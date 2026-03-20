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
    S_IDLE    = 4'd0,
    S_LOAD_W  = 4'd1,
    S_LOAD_A  = 4'd2,
    S_READY   = 4'd3,
    S_COMPUTE = 4'd4,
    S_FLUSH   = 4'd5,
    S_FENCE   = 4'd6;

reg [3:0]  state;
reg [7:0]  cfg_Tk_r;
reg [8:0]  cfg_Tk_full;     // 9-bit: 真实 Tk 值 (1~256)
reg [7:0]  num_a_chunks_r;  // cfg_Tk_full >> 4, MCFG 时缓存

reg signed [7:0] wbuf [0:MAX_TK-1][0:15];
reg signed [7:0] abuf [0:15][0:MAX_TK-1];

reg [7:0]  load_cnt;      // MLOAD_W: k-row 计数器
reg [3:0]  a_row;          // MLOAD_A: 当前行 (0-15)
reg [7:0]  a_chunk;        // MLOAD_A: 当前行内的 chunk
reg [8:0]  k_cnt;
reg [5:0]  flush_cnt;

assign act_ready = (state == S_LOAD_A);

// -------------------------------------------------------------
// 16x16 PE 阵列
// -------------------------------------------------------------
wire        pe_en;
wire        pe_acc_clear;
wire signed [31:0] pe_acc_out [0:15][0:15];

assign pe_acc_clear = (op_valid && op_type == OP_MZERO && (state == S_IDLE || state == S_READY));
assign pe_en        = (state == S_COMPUTE && k_cnt < cfg_Tk_full);

genvar gm, gn;
generate
    for (gm = 0; gm < 16; gm = gm + 1) begin : g_pe_m
        for (gn = 0; gn < 16; gn = gn + 1) begin : g_pe_n
            rvv_backend_mxu_pe u_pe (
                .clk       (clk),
                .rst_n     (rst_n),
                .en        (pe_en),
                .acc_clear (pe_acc_clear),
                .act_in    (abuf[gm][k_cnt[7:0]]),
                .weight_in (wbuf[k_cnt[7:0]][gn]),
                .acc_out   (pe_acc_out[gm][gn])
            );
        end
    end
endgenerate

integer j;

// -------------------------------------------------------------
// MSTORE 输出数据组合逻辑
// -------------------------------------------------------------
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

// 辅助任务: 推进 a_row/a_chunk 计数器
task automatic advance_a_counters;
    if (a_chunk == num_a_chunks_r - 8'd1) begin
        a_chunk <= 8'd0;
        a_row   <= a_row + 4'd1;
    end else begin
        a_chunk <= a_chunk + 8'd1;
    end
endtask

// 辅助任务: 将一个 activation beat 写入 abuf
task automatic write_abuf_beat;
    for (j = 0; j < 16; j = j + 1)
        abuf[a_row][(a_chunk << 4) + j] <= $signed(act_vec[j*8 +: 8]);
endtask

// -------------------------------------------------------------
// 主状态机
// -------------------------------------------------------------
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
                    // 0 表示 256
                    cfg_Tk_full    <= (cfg_Tk == 8'd0) ? 9'd256 : {1'b0, cfg_Tk};
                    // chunks per row: 256>>4=16, 其他正常计算
                    num_a_chunks_r <= (cfg_Tk == 8'd0) ? 8'd16
                                   : (cfg_Tk[7:4] == 4'd0) ? 8'd1
                                   : {4'd0, cfg_Tk[7:4]};
                    op_done        <= 1'b1;
                end

                OP_MZERO: begin
                    op_done <= 1'b1;
                end

                OP_MLOAD_W: begin
                    for (j = 0; j < 16; j = j + 1)
                        wbuf[load_cnt][j] <= $signed(weight_vec[j*8 +: 8]);

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
                    write_abuf_beat;
                    if (uop_last) begin
                        op_done  <= 1'b1;
                        op_ready <= 1'b1;
                        a_row    <= 4'd0;
                        a_chunk  <= 8'd0;
                    end else begin
                        advance_a_counters;
                        op_ready <= 1'b0;
                        state    <= S_LOAD_A;
                    end
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

        S_LOAD_W: begin
            result_valid <= 1'b0;
            if (weight_valid) begin
                for (j = 0; j < 16; j = j + 1)
                    wbuf[load_cnt][j] <= $signed(weight_vec[j*8 +: 8]);

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

        S_LOAD_A: begin
            result_valid <= 1'b0;
            if (act_valid) begin
                write_abuf_beat;
                if (uop_last) begin
                    op_done  <= 1'b1;
                    op_ready <= 1'b1;
                    a_row    <= 4'd0;
                    a_chunk  <= 8'd0;
                    state    <= S_READY;
                end else begin
                    advance_a_counters;
                end
            end
        end

        S_COMPUTE: begin
            result_valid <= 1'b0;
            if (k_cnt < cfg_Tk_full) begin
                k_cnt <= k_cnt + 9'd1;
            end else begin
                op_done  <= 1'b1;
                op_ready <= 1'b1;
                state    <= S_READY;
            end
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