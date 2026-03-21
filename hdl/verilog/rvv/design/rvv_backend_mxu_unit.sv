module rvv_backend_mxu_unit (
    input  wire        clk,
    input  wire        rst_n,

    input  wire [2:0]  op_type,
    input  wire        op_valid,
    output reg         op_ready,
    output reg         op_done,

    input  wire [5:0]  uop_index,
    input  wire        uop_last,

    input  wire [7:0]  cfg_Tk,
    input  wire        cfg_signed,

    input  wire [127:0] weight_vec,
    input  wire         weight_valid,
    output reg          weight_ready,

    input  wire [127:0] act_vec,
    input  wire         act_valid,
    output wire         act_ready,

    output reg  [127:0] result_data,
    output reg          result_valid,
    input  wire         result_ready,
    output reg          mfence_done
);

localparam MAX_TK = 256;

localparam [2:0]
    OP_MCFG    = 3'd0,
    OP_MLOAD_W = 3'd1,
    OP_MLOAD_A = 3'd2,
    OP_MZERO   = 3'd3,
    OP_MMA     = 3'd4,
    OP_MSTORE  = 3'd5,
    OP_MFENCE  = 3'd6;

localparam [3:0]
    S_IDLE      = 4'd0,
    S_LOAD_W    = 4'd1,
    S_LOAD_A    = 4'd2,
    S_READY     = 4'd3,
    S_COMPUTE   = 4'd4,
    S_COMP_RD   = 4'd5,
    S_FLUSH     = 4'd6,
    S_FENCE     = 4'd7,
    S_LOAD_A_EX = 4'd8;  // MLOAD_A 展开写入状态

reg [3:0]  state;
reg [7:0]  cfg_Tk_r;
reg [8:0]  cfg_Tk_full;
reg [7:0]  num_a_chunks_r;

// ---------------------------------------------------------------
// Weight SRAM
// ---------------------------------------------------------------
reg          wbuf_en, wbuf_wr;
reg  [7:0]   wbuf_addr;
reg  [127:0] wbuf_wdata;
wire [127:0] wbuf_rdata;

Sram_256x128 u_wbuf (
    .clock (clk),
    .enable(wbuf_en),
    .write (wbuf_wr),
    .addr  (wbuf_addr),
    .wdata (wbuf_wdata),
    .wmask (16'hFFFF),
    .rdata (wbuf_rdata)
);

// ---------------------------------------------------------------
// Activation SRAM (转置存储)
// 地址 k -> {abuf[15][k], ..., abuf[1][k], abuf[0][k]}
// ---------------------------------------------------------------
reg          abuf_en, abuf_wr;
reg  [7:0]   abuf_addr;
reg  [127:0] abuf_wdata;
reg  [15:0]  abuf_wmask;
wire [127:0] abuf_rdata;

Sram_256x128 u_abuf (
    .clock (clk),
    .enable(abuf_en),
    .write (abuf_wr),
    .addr  (abuf_addr),
    .wdata (abuf_wdata),
    .wmask (abuf_wmask),
    .rdata (abuf_rdata)
);

reg [7:0]  load_cnt;
reg [3:0]  a_row;
reg [7:0]  a_chunk;
reg [8:0]  k_cnt;
reg [5:0]  flush_cnt;

// MLOAD_A 展开写入相关
reg [127:0] a_beat_buf;       // 缓存当前 beat 的 128bit 数据
reg [3:0]   a_expand_cnt;     // 展开计数 0..15
reg         a_beat_is_last;   // 当前 beat 是否是最后一拍
reg [3:0]   a_row_saved;      // 展开期间保存的 row
reg [7:0]   a_chunk_saved;    // 展开期间保存的 chunk

// ---------------------------------------------------------------
// SRAM read pipeline (1 cycle latency)
// ---------------------------------------------------------------
wire sram_rd_now_w = (wbuf_en && !wbuf_wr);
wire sram_rd_now_a = (abuf_en && !abuf_wr);

reg        w_rd_valid_p1;
reg        a_rd_valid_p1;
reg [8:0]  k_cnt_p1;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        w_rd_valid_p1 <= 1'b0;
        a_rd_valid_p1 <= 1'b0;
        k_cnt_p1      <= 9'd0;
    end else begin
        w_rd_valid_p1 <= sram_rd_now_w;
        a_rd_valid_p1 <= sram_rd_now_a;
        k_cnt_p1      <= k_cnt;
    end
end

// 两个 SRAM 同时读，都在 1 拍后有效
// PE enable: 两者都有效时才使能
wire pe_data_valid = w_rd_valid_p1 & a_rd_valid_p1;

assign act_ready = (state == S_LOAD_A);

// ---------------------------------------------------------------
// PE weight input: 直接从 wbuf_rdata 组合取
// ---------------------------------------------------------------
wire signed [7:0] pe_weight_in [0:15];
genvar gi;
generate
    for (gi = 0; gi < 16; gi = gi + 1) begin : g_wt
        assign pe_weight_in[gi] = $signed(wbuf_rdata[gi*8 +: 8]);
    end
endgenerate

// ---------------------------------------------------------------
// PE act input: 直接从 abuf_rdata 组合取
// abuf_rdata[row*8 +: 8] = abuf[row][k]
// ---------------------------------------------------------------
wire signed [7:0] pe_act_in [0:15];
genvar gm, gn;
generate
    for (gm = 0; gm < 16; gm = gm + 1) begin : g_act
        assign pe_act_in[gm] = $signed(abuf_rdata[gm*8 +: 8]);
    end
endgenerate

// ---------------------------------------------------------------
// 16x16 PE array
// ---------------------------------------------------------------
wire        pe_en;
wire        pe_acc_clear;
wire signed [31:0] pe_acc_out [0:15][0:15];

assign pe_acc_clear = (op_valid && op_type == OP_MZERO &&
                       (state == S_IDLE || state == S_READY));
assign pe_en        = pe_data_valid;

generate
    for (gm = 0; gm < 16; gm = gm + 1) begin : g_pe_m
        for (gn = 0; gn < 16; gn = gn + 1) begin : g_pe_n
            rvv_backend_mxu_pe u_pe (
                .clk       (clk),
                .rst_n     (rst_n),
                .en        (pe_en),
                .acc_clear (pe_acc_clear),
                .act_in    (pe_act_in[gm]),
                .weight_in (pe_weight_in[gn]),
                .acc_out   (pe_acc_out[gm][gn])
            );
        end
    end
endgenerate

integer j;

// ---------------------------------------------------------------
// MSTORE output mux
// ---------------------------------------------------------------
reg [127:0] flush_data_comb;
always @(*) begin
    case (flush_cnt[1:0])
        2'd0: flush_data_comb = {pe_acc_out[flush_cnt[5:2]][ 3],
                                  pe_acc_out[flush_cnt[5:2]][ 2],
                                  pe_acc_out[flush_cnt[5:2]][ 1],
                                  pe_acc_out[flush_cnt[5:2]][ 0]};
        2'd1: flush_data_comb = {pe_acc_out[flush_cnt[5:2]][ 7],
                                  pe_acc_out[flush_cnt[5:2]][ 6],
                                  pe_acc_out[flush_cnt[5:2]][ 5],
                                  pe_acc_out[flush_cnt[5:2]][ 4]};
        2'd2: flush_data_comb = {pe_acc_out[flush_cnt[5:2]][11],
                                  pe_acc_out[flush_cnt[5:2]][10],
                                  pe_acc_out[flush_cnt[5:2]][ 9],
                                  pe_acc_out[flush_cnt[5:2]][ 8]};
        2'd3: flush_data_comb = {pe_acc_out[flush_cnt[5:2]][15],
                                  pe_acc_out[flush_cnt[5:2]][14],
                                  pe_acc_out[flush_cnt[5:2]][13],
                                  pe_acc_out[flush_cnt[5:2]][12]};
    endcase
end

// ---------------------------------------------------------------
// abuf SRAM 组合控制
// ---------------------------------------------------------------
always @(*) begin
    abuf_en    = 1'b0;
    abuf_wr    = 1'b0;
    abuf_addr  = 8'd0;
    abuf_wdata = 128'd0;
    abuf_wmask = 16'h0000;

    case (state)
    S_LOAD_A_EX: begin
        // 展开写入：每周期写 1 个 SRAM 地址，只更新第 a_row_saved 个 byte
        abuf_en    = 1'b1;
        abuf_wr    = 1'b1;
        abuf_addr  = (a_chunk_saved << 4) | {4'd0, a_expand_cnt};
        // 把当前 byte 复制到 wdata 的第 a_row_saved 个 byte 位置
        abuf_wdata = {16{a_beat_buf[a_expand_cnt*8 +: 8]}};
        abuf_wmask = 16'd1 << a_row_saved;
    end
    S_COMPUTE: begin
        if (k_cnt < cfg_Tk_full) begin
            abuf_en   = 1'b1;
            abuf_wr   = 1'b0;
            abuf_addr = k_cnt[7:0];
        end
    end
    default: ;
    endcase
end

// ---------------------------------------------------------------
// wbuf SRAM 组合控制
// ---------------------------------------------------------------
always @(*) begin
    wbuf_en    = 1'b0;
    wbuf_wr    = 1'b0;
    wbuf_addr  = 8'd0;
    wbuf_wdata = 128'd0;

    case (state)
    S_IDLE, S_READY: begin
        if (op_valid && op_type == OP_MLOAD_W) begin
            wbuf_en    = 1'b1;
            wbuf_wr    = 1'b1;
            wbuf_addr  = load_cnt;
            wbuf_wdata = weight_vec;
        end
    end
    S_LOAD_W: begin
        if (weight_valid) begin
            wbuf_en    = 1'b1;
            wbuf_wr    = 1'b1;
            wbuf_addr  = load_cnt;
            wbuf_wdata = weight_vec;
        end
    end
    S_COMPUTE: begin
        if (k_cnt < cfg_Tk_full) begin
            wbuf_en   = 1'b1;
            wbuf_wr   = 1'b0;
            wbuf_addr = k_cnt[7:0];
        end
    end
    default: ;
    endcase
end

// ---------------------------------------------------------------
// 主状态机
// ---------------------------------------------------------------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state           <= S_IDLE;
        op_ready        <= 1'b1;
        op_done         <= 1'b0;
        weight_ready    <= 1'b0;
        result_valid    <= 1'b0;
        mfence_done     <= 1'b0;
        load_cnt        <= 8'd0;
        a_row           <= 4'd0;
        a_chunk         <= 8'd0;
        num_a_chunks_r  <= 8'd4;
        k_cnt           <= 9'd0;
        flush_cnt       <= 6'd0;
        cfg_Tk_r        <= 8'd64;
        cfg_Tk_full     <= 9'd64;
        result_data     <= 128'd0;
        a_beat_buf      <= 128'd0;
        a_expand_cnt    <= 4'd0;
        a_beat_is_last  <= 1'b0;
        a_row_saved     <= 4'd0;
        a_chunk_saved   <= 8'd0;
    end else begin
        op_done      <= 1'b0;
        mfence_done  <= 1'b0;

        case (state)
        S_IDLE, S_READY: begin
            op_ready     <= 1'b1;
            weight_ready <= 1'b0;
            result_valid <= 1'b0;

            if (op_valid) begin
                case (op_type)
                OP_MCFG: begin
                    cfg_Tk_r       <= cfg_Tk;
                    cfg_Tk_full    <= (cfg_Tk == 8'd0) ? 9'd256 : {1'b0, cfg_Tk};
                    num_a_chunks_r <= (cfg_Tk == 8'd0) ? 8'd16
                                   : (cfg_Tk[7:4] == 4'd0) ? 8'd1
                                   : {4'd0, cfg_Tk[7:4]};
                    op_done        <= 1'b1;
                end

                OP_MZERO: begin
                    op_done <= 1'b1;
                end

                OP_MLOAD_W: begin
                    if (uop_last) begin
                        op_done  <= 1'b1;
                        load_cnt <= 8'd0;
                    end else begin
                        load_cnt     <= load_cnt + 8'd1;
                        op_ready     <= 1'b0;
                        weight_ready <= 1'b1;
                        state        <= S_LOAD_W;
                    end
                end

                OP_MLOAD_A: begin
                    // 缓存 beat 数据，进入展开写入状态
                    a_beat_buf     <= act_vec;
                    a_row_saved    <= a_row;
                    a_chunk_saved  <= a_chunk;
                    a_beat_is_last <= uop_last;
                    a_expand_cnt   <= 4'd0;
                    op_ready       <= 1'b0;
                    state          <= S_LOAD_A_EX;
                end

                OP_MMA: begin
                    op_ready <= 1'b0;
                    k_cnt    <= 9'd0;
                    state    <= S_COMPUTE;
                end

                OP_MSTORE: begin
                    result_valid <= 1'b1;
                    result_data  <= flush_data_comb;
                    op_done      <= 1'b1;
                    if (flush_cnt == 6'd63)
                        flush_cnt <= 6'd0;
                    else
                        flush_cnt <= flush_cnt + 6'd1;
                end

                OP_MFENCE: begin
                    mfence_done <= 1'b1;
                    op_done     <= 1'b1;
                end
                endcase
            end
        end

        // -------------------------------------------------------
        // MLOAD_A 展开写入：16 周期把 1 个 beat 写入 abuf SRAM
        // -------------------------------------------------------
        S_LOAD_A_EX: begin
            result_valid <= 1'b0;
            if (a_expand_cnt == 4'd15) begin
                // 16 个字节全部写完
                if (a_beat_is_last) begin
                    // 整个 MLOAD_A 指令结束
                    op_done  <= 1'b1;
                    op_ready <= 1'b1;
                    a_row    <= 4'd0;
                    a_chunk  <= 8'd0;
                    state    <= S_READY;
                end else begin
                    // 推进 row/chunk 计数器，回到 S_LOAD_A 等下一个 beat
                    if (a_chunk == num_a_chunks_r - 8'd1) begin
                        a_chunk <= 8'd0;
                        a_row   <= a_row + 4'd1;
                    end else begin
                        a_chunk <= a_chunk + 8'd1;
                    end
                    op_ready <= 1'b0;
                    state    <= S_LOAD_A;
                end
            end else begin
                a_expand_cnt <= a_expand_cnt + 4'd1;
            end
        end

        // -------------------------------------------------------
        // S_LOAD_A: 等待下一个 act beat
        // -------------------------------------------------------
        S_LOAD_A: begin
            result_valid <= 1'b0;
            if (act_valid) begin
                // 缓存并进入展开
                a_beat_buf     <= act_vec;
                a_row_saved    <= a_row;
                a_chunk_saved  <= a_chunk;
                a_beat_is_last <= uop_last;
                a_expand_cnt   <= 4'd0;
                state          <= S_LOAD_A_EX;
            end
        end

        S_LOAD_W: begin
            result_valid <= 1'b0;
            if (weight_valid) begin
                if (uop_last) begin
                    weight_ready <= 1'b0;
                    op_done      <= 1'b1;
                    op_ready     <= 1'b1;
                    load_cnt     <= 8'd0;
                    state        <= S_READY;
                end else begin
                    load_cnt <= load_cnt + 8'd1;
                end
            end
        end

        S_COMPUTE: begin
            result_valid <= 1'b0;
            if (k_cnt < cfg_Tk_full) begin
                k_cnt <= k_cnt + 9'd1;
            end else begin
                if (!pe_data_valid) begin
                    state <= S_COMP_RD;
                end
            end
        end

        S_COMP_RD: begin
            // 等 PE 第二级（累加）完成
            op_done  <= 1'b1;
            op_ready <= 1'b1;
            state    <= S_READY;
        end

        S_FLUSH: begin
            op_ready <= 1'b1;
            state    <= S_READY;
        end

        S_FENCE: begin
            result_valid <= 1'b0;
            mfence_done  <= 1'b1;
            op_done      <= 1'b1;
            op_ready     <= 1'b1;
            state        <= S_IDLE;
        end

        default: begin
            state    <= S_IDLE;
            op_ready <= 1'b1;
        end
        endcase
    end
end

endmodule