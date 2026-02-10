`ifndef HDL_VERILOG_RVV_DESIGN_RVV_SVH
`include "rvv_backend.svh"
`endif
`ifndef RVV_ASSERT__SVH
`include "rvv_backend_sva.svh"
`endif

 module rvv_backend_mxu (
    input   logic                   clk,
    input   logic                   rst_n,

    // 1. 来自 RS 的握手和数据
    input   logic                   pop_ex2rs,         // 确认信号
    input   MXU_RS_t                mxu_uop_rs2ex,     // 操作数内容 (源数据在这里)
    input   logic                   fifo_empty_rs2ex,
    input   logic                   fifo_almost_empty_rs2ex,

    // 2. 写回 ROB 的结果
    output  logic                   result_valid_ex2rob,
    output  PU2ROB_t                result_ex2rob,     // 计算结果
    input   logic                   result_ready_rob2mxu, // ROB是否准备好接收

    // 3. 全局控制
    input   logic                   trap_flush_rvv
);




endmodule
