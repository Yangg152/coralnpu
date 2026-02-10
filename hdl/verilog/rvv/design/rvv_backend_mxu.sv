`ifndef HDL_VERILOG_RVV_DESIGN_RVV_SVH
`include "rvv_backend.svh"
`endif
`ifndef RVV_ASSERT__SVH
`include "rvv_backend_sva.svh"
`endif

module rvv_backend_mxu (
  // Global Signals
  input  logic                   clk,
  input  logic                   rst_n,
  
  // ========================================================================
  // 1. Frontend Interface: From Reservation Station (RS) / Dispatch
  // ========================================================================
  // 握手信号：告诉 RS 我准备好接收下一条指令了 (Pop)
  output logic                   pop_ex2rs,        
  
  // 数据信号：RS 发送过来的完整微指令包
  // 包含：vs1_data (Activation), vs2_data (Accumulator), subop, route_mode, mode_wide, weight_idx 等
  input  MXU_RS_t                mxu_uop_rs2ex,    
  
  // 状态信号：RS FIFO 是否为空 (如果为空，MXU 应该待机)
  input  logic                   fifo_empty_rs2ex, 

  // ========================================================================
  // 2. Backend Interface: To Reorder Buffer (ROB) / Writeback
  // ========================================================================
  // 握手信号：ROB 是否准备好接收结果 (Backpressure)
  input  logic                   result_ready_rob2mxu,
  
  // 有效信号：MXU 计算完成，结果有效
  output logic                   result_valid_ex2rob, 
  
  // 结果信号：写回 VRF 的数据包
  // 通常包含：w_data (结果数据), w_index (目标寄存器索引), w_valid, vsaturate 等
  output RES_ROB_t               result_ex2rob,    

  // ========================================================================
  // 3. Control & Exception Interface
  // ========================================================================
  // 全局刷新信号：发生异常或预测失败时，清空 MXU 内部流水线和状态机
  input  logic                   trap_flush_rvv
);

  // ========================================================================
  // Internal Signals & Logic Placeholders (后续需要实现的逻辑)
  // ========================================================================

  // 1. Pipeline Control Logic
  // -------------------------
  // 状态机：IDLE -> LOAD_WEIGHT -> COMPUTE -> WRITEBACK
  logic mxu_busy;
  
  // 2. Weight Buffer (SRAM)
  // -------------------------
  // 4KB SRAM, used when subop == MXU_WLOAD or MXU_MMA (read)
  // logic [7:0] weight_mem [0:4095]; 

  // 3. Lightweight Operand Router (Mux Network)
  // -------------------------
  // 根据 mxu_uop_rs2ex.route_mode 和 mode_wide 对 vs1_data 进行重排
  // logic [8*16-1:0] systolic_array_inputs [0:15];

  // 4. Systolic Array Core (16x16)
  // -------------------------
  // 256 个 MAC 单元
  
  // 5. Output Collector
  // -------------------------
  // 处理 Deep Mode (16 outputs) vs Wide Mode (4 outputs x 4) 的结果收集

  // ------------------------------------------------------------------------
  // Simple Handshake Logic (Example)
  // 只有当 RS 有数据 (非空)，且 MXU 不忙，且 ROB 准备好接收时，才 Pop
  // assign pop_ex2rs = !fifo_empty_rs2ex && !mxu_busy && result_ready_rob2mxu;

endmodule
