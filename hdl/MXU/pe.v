`timescale 1ns / 1ps
module pe (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,
    input  wire        acc_clear,
    input  wire signed [7:0]  act_in,
    input  wire signed [7:0]  weight_in,
    output wire signed [31:0] acc_out
);

    // Stage 1: 乘法（INT8×INT8 → INT16），寄存
    reg signed [15:0] mul_reg;
    reg               en_d1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mul_reg <= 16'sd0;
            en_d1   <= 1'b0;
        end else begin
            en_d1 <= en;
            if (en)
                mul_reg <= act_in * weight_in;
            else
                mul_reg <= 16'sd0;
        end
    end

    // Stage 2: 累加（INT16符号扩展 → INT32）
    reg signed [31:0] acc_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_reg <= 32'sd0;
        end else if (acc_clear) begin
            acc_reg <= 32'sd0;
        end else if (en_d1) begin
            acc_reg <= acc_reg + $signed({{16{mul_reg[15]}}, mul_reg});
        end
    end

    assign acc_out = acc_reg;

endmodule
