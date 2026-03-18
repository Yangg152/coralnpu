`timescale 1ns / 1ps
module mxu (
    input  wire        clk,
    input  wire        rst_n,

    input  wire [2:0]  op_type,
    input  wire        op_valid,
    output reg         op_ready,
    output reg         op_done,

    input  wire [7:0]  cfg_Tk,       
    input  wire        cfg_signed,

    input  wire [127:0] weight_vec,
    input  wire         weight_valid,
    output reg          weight_ready,

    input  wire [127:0] act_row [0:15],
    input  wire         act_valid,
    output wire         act_ready,

    output reg  [127:0] result_bus [0:15], 
    output reg          result_valid,
    output reg          mfence_done
);

localparam MAX_TK = 128;

localparam [2:0]
    OP_MCFG    = 3'd0,
    OP_MLOAD_W = 3'd1,
    OP_MZERO   = 3'd2,
    OP_MMA     = 3'd3,
    OP_MSTORE  = 3'd4,
    OP_MFENCE  = 3'd5,
    OP_MLOAD_A = 3'd6; 

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

// 存储结构
reg signed [7:0] wbuf [0:MAX_TK-1][0:15];
reg signed [7:0] abuf [0:15][0:MAX_TK-1];  

reg [7:0]  load_cnt;
reg [8:0]  k_cnt;
reg [2:0]  flush_cnt; 

assign act_ready = (state == S_LOAD_A);

// -------------------------------------------------------------
// 16x16 PE 阵列例化与信号分发
// -------------------------------------------------------------
wire        pe_en;
wire        pe_acc_clear;
wire signed [31:0] pe_acc_out [0:15][0:15];

// 清零信号：当为 MZERO 时，发给所有 PE 的累加器
assign pe_acc_clear = (op_valid && op_type == OP_MZERO && (state == S_IDLE || state == S_READY));
// PE 使能：在计算状态下，发出 k_cnt 的激活和权重
assign pe_en        = (state == S_COMPUTE && k_cnt < {1'b0, cfg_Tk_r});

genvar m, n;
generate
    for (m = 0; m < 16; m = m + 1) begin : g_pe_m
        for (n = 0; n < 16; n = n + 1) begin : g_pe_n
            pe u_pe (
                .clk       (clk),
                .rst_n     (rst_n),
                .en        (pe_en),
                .acc_clear (pe_acc_clear),
                .act_in    (abuf[m][k_cnt[6:0]]), // 广播
                .weight_in (wbuf[k_cnt[6:0]][n]), // 驻留
                .acc_out   (pe_acc_out[m][n])
            );
        end
    end
endgenerate

integer i, j;

// -------------------------------------------------------------
// 主状态机
// -------------------------------------------------------------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state        <= S_IDLE;
        op_ready     <= 1'b1;
        op_done      <= 1'b0;
        weight_ready <= 1'b0;
        result_valid <= 1'b0;
        mfence_done  <= 1'b0;
        load_cnt     <= 8'd0;
        k_cnt        <= 9'd0;
        flush_cnt    <= 3'd0;
        cfg_Tk_r     <= 8'd64;

        for (i=0; i<16; i=i+1) result_bus[i] <= 128'd0;
    end else begin
        op_done      <= 1'b0;
        result_valid <= 1'b0;
        mfence_done  <= 1'b0;

        case (state)
        S_IDLE, S_READY: begin
            op_ready     <= 1'b1;
            weight_ready <= 1'b0;
            if (op_valid) begin
                case (op_type)
                OP_MCFG: begin
                    cfg_Tk_r <= cfg_Tk;
                    op_done  <= 1'b1;
                end
                OP_MZERO: begin
                    op_done <= 1'b1; // 清零在组合逻辑 pe_acc_clear 被捕获
                end
                OP_MLOAD_W: begin
                    load_cnt     <= 8'd0;
                    op_ready     <= 1'b0;
                    weight_ready <= 1'b1;
                    state        <= S_LOAD_W;
                end
                OP_MLOAD_A: begin
                    load_cnt <= 8'd0;
                    op_ready <= 1'b0;
                    state    <= S_LOAD_A;
                end
                OP_MMA: begin
                    op_ready <= 1'b0;
                    k_cnt    <= 9'd0;
                    state    <= S_COMPUTE;
                end
                OP_MSTORE: begin
                    op_ready  <= 1'b0;
                    flush_cnt <= 3'd0;
                    state     <= S_FLUSH;
                end
                OP_MFENCE: begin
                    op_ready <= 1'b0;
                    state    <= S_FENCE;
                end
                endcase
            end
        end

        S_LOAD_W: begin
            if (weight_valid) begin
                for (j = 0; j < 16; j = j + 1)
                    wbuf[load_cnt][j] <= $signed(weight_vec[j*8 +: 8]);

                if (load_cnt == cfg_Tk_r - 8'd1) begin
                    weight_ready <= 1'b0;
                    op_done      <= 1'b1;
                    op_ready     <= 1'b1;
                    state        <= S_READY;
                end else begin
                    load_cnt <= load_cnt + 8'd1;
                end
            end
        end

        S_LOAD_A: begin
            if (act_valid) begin
                for (i = 0; i < 16; i = i + 1)
                    for (j = 0; j < 16; j = j + 1)
                        abuf[i][(load_cnt<<4) + j] <= $signed(act_row[i][j*8 +: 8]);

                // 需要装载 Tk / 16 次
                if (load_cnt == (cfg_Tk_r >> 4) - 8'd1) begin
                    op_done  <= 1'b1;
                    op_ready <= 1'b1;
                    state    <= S_READY;
                end else begin
                    load_cnt <= load_cnt + 8'd1;
                end
            end
        end

        S_COMPUTE: begin
            // k_cnt == Tk_r 时，pe_en变0，额外等1拍让加法器落盘
            if (k_cnt < {1'b0, cfg_Tk_r}) begin
                k_cnt <= k_cnt + 9'd1;
            end else begin
                op_done  <= 1'b1;
                op_ready <= 1'b1;
                state    <= S_READY;
            end
        end

        S_FLUSH: begin
            result_valid <= 1'b1;
            // 每拍输出4行结果，共16条128bit总线
            begin : flush_blk
                integer r, c;
                for (r = 0; r < 4; r = r + 1) begin
                    for (c = 0; c < 4; c = c + 1) begin
                        result_bus[r*4 + c] <= {
                            pe_acc_out[flush_cnt*4 + r][c*4 + 3],
                            pe_acc_out[flush_cnt*4 + r][c*4 + 2],
                            pe_acc_out[flush_cnt*4 + r][c*4 + 1],
                            pe_acc_out[flush_cnt*4 + r][c*4 + 0]
                        };
                    end
                end
            end

            if (flush_cnt == 3'd3) begin // 满4拍结束
                op_done  <= 1'b1;
                op_ready <= 1'b1;
                state    <= S_READY;
            end else begin
                flush_cnt <= flush_cnt + 3'd1;
            end
        end

        S_FENCE: begin
            mfence_done <= 1'b1;
            op_done     <= 1'b1;
            op_ready    <= 1'b1;
            state       <= S_IDLE;
        end
        endcase
    end
end
endmodule
