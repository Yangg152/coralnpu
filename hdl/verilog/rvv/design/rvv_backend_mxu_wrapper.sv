// file: design/rvv_backend_mxu_wrapper.sv
`ifndef HDL_VERILOG_RVV_DESIGN_RVV_SVH
`include "rvv_backend.svh"
`endif

module rvv_backend_mxu_wrapper (
    input  logic                            clk,
    input  logic                            rst_n,

    output logic        [`NUM_MXU-1:0]      pop_ex2rs,
    input  MXU_RS_t     [`NUM_MXU-1:0]      mxu_uop_rs2ex,
    input  logic                            fifo_empty_rs2ex,
    input  logic        [`NUM_MXU-1:0]      fifo_almost_empty_rs2ex,

    output logic        [`NUM_MXU-1:0]      result_valid_ex2rob,
    output PU2ROB_t     [`NUM_MXU-1:0]      result_ex2rob,
    input  logic        [`NUM_MXU-1:0]      result_ready_rob2ex,

    input  logic                            trap_flush_rvv
);

    // ---------------------------------------------------------------
    // MXU core signals
    // ---------------------------------------------------------------
    wire        mxu_op_ready;
    wire        mxu_op_done;
    wire        mxu_weight_ready;
    wire        mxu_act_ready;
    wire        mxu_result_valid;
    wire        mxu_mfence_done;
    wire [127:0] mxu_result_data;

    // ---------------------------------------------------------------
    // Decode current uop fields from RS head
    // ---------------------------------------------------------------
    wire [2:0]   op_type    = mxu_uop_rs2ex[0].uop_funct6[2:0];
    wire [7:0]   cfg_Tk     = mxu_uop_rs2ex[0].rs1_data[7:0];
    wire         cfg_signed = mxu_uop_rs2ex[0].rs1_data[8];
    wire [127:0] weight_vec = mxu_uop_rs2ex[0].vs2_data[127:0];
    wire [127:0] act_vec    = mxu_uop_rs2ex[0].vs2_data[127:0];

    wire [5:0]   row_index      = mxu_uop_rs2ex[0].rs1_data[5:0];
    wire         uop_last_from_vm = ~mxu_uop_rs2ex[0].vm;

    // ---------------------------------------------------------------
    // RS has data?
    // ---------------------------------------------------------------
    wire rs_has_data = !fifo_empty_rs2ex;

    // ---------------------------------------------------------------
    // Pop logic: accept uop when MXU unit can take it
    //
    // For MLOAD_W in S_LOAD_W state, MXU unit asserts weight_ready.
    // For MLOAD_A in S_LOAD_A state, MXU unit asserts act_ready.
    // We must pop from RS in those states too.
    // ---------------------------------------------------------------
    logic can_accept;
    always_comb begin
        can_accept = 1'b0;
        if (rs_has_data) begin
            case (op_type)
                3'd1:    can_accept = mxu_op_ready | mxu_weight_ready; // MLOAD_W
                3'd2:    can_accept = mxu_op_ready | mxu_act_ready;    // MLOAD_A
                default: can_accept = mxu_op_ready;
            endcase
        end
    end

    assign pop_ex2rs[0] = can_accept;

    // ---------------------------------------------------------------
    // Valid signals to MXU unit
    //
    // op_valid: asserted on the FIRST beat of each instruction
    //           (when mxu_op_ready accepts it)
    // weight_valid: asserted for ALL MLOAD_W beats (first + continuation)
    // act_valid: asserted for ALL MLOAD_A beats
    // ---------------------------------------------------------------
    wire first_beat   = pop_ex2rs[0] & mxu_op_ready;
    wire weight_beat  = pop_ex2rs[0] & (op_type == 3'd1);
    wire act_beat     = pop_ex2rs[0] & (op_type == 3'd2);

    wire op_valid     = first_beat;
    wire weight_valid = weight_beat;
    wire act_valid    = act_beat;

    // ---------------------------------------------------------------
    // MXU core instance
    // ---------------------------------------------------------------
    rvv_backend_mxu_unit u_mxu_core (
        .clk            (clk),
        .rst_n          (rst_n & ~trap_flush_rvv),

        .op_type        (op_type),
        .op_valid       (op_valid),
        .op_ready       (mxu_op_ready),
        .op_done        (mxu_op_done),

        .uop_index      (row_index),
        .uop_last       (uop_last_from_vm),

        .cfg_Tk         (cfg_Tk),
        .cfg_signed     (cfg_signed),

        .weight_vec     (weight_vec),
        .weight_valid   (weight_valid),
        .weight_ready   (mxu_weight_ready),

        .act_vec        (act_vec),
        .act_valid      (act_valid),
        .act_ready      (mxu_act_ready),

        .result_data    (mxu_result_data),
        .result_valid   (mxu_result_valid),
        .result_ready   (result_ready_rob2ex[0]),
        .mfence_done    (mxu_mfence_done)
    );

    // ---------------------------------------------------------------
    // ROB writeback
    //
    // Every popped uop must produce a ROB writeback so the ROB entry
    // gets marked done.
    //
    // Pipeline by 1 cycle for pop/rob_entry/op_type only.
    // For MSTORE data: read mxu_result_data directly (combinational)
    // at the cycle when pop_d1 is asserted, because MXU unit's
    // result_data register has been updated by then.
    //
    // Timing:
    //   T0: pop MSTORE, MXU writes result_data <= flush_data_comb
    //   T1: result_data is valid, pop_d1=1, rob_entry_d1=correct
    //       -> read mxu_result_data directly = correct data
    // ---------------------------------------------------------------
    logic                          pop_d1;
    logic [`ROB_DEPTH_WIDTH-1:0]   rob_entry_d1;
    logic [2:0]                    op_type_d1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n || trap_flush_rvv) begin
            pop_d1         <= 1'b0;
            rob_entry_d1   <= '0;
            op_type_d1     <= '0;
        end else begin
            pop_d1         <= pop_ex2rs[0];
            rob_entry_d1   <= mxu_uop_rs2ex[0].rob_entry;
            op_type_d1     <= op_type;
        end
    end

    assign result_valid_ex2rob[0] = pop_d1;

    always_comb begin
        result_ex2rob[0]           = '0;
        result_ex2rob[0].rob_entry = rob_entry_d1;
        if (op_type_d1 == 3'd5) begin  // MSTORE
            result_ex2rob[0].w_valid = 1'b1;
            result_ex2rob[0].w_data  = {{(`VLEN-128){1'b0}}, mxu_result_data};
        end else begin
            result_ex2rob[0].w_valid = 1'b0;
        end
        result_ex2rob[0].vsaturate = '0;
    end
    
    // initial begin
    //     $display("[MXU Wrapper] Module instantiated and simulation started!");
    // end
    
    // // 在 rvv_backend_mxu_wrapper.sv 的 always block 或 initial block 附近加：
    // always @(posedge clk) begin
    //     if (pop_ex2rs[0]) begin
    //         $display("[MXU-WRAP] T=%0t pop! op=%0d rs1_data=0x%h rs1_valid=%b funct6=%b funct3=%b vm=%b",
    //                 $time, op_type,
    //                 mxu_uop_rs2ex[0].rs1_data,
    //                 mxu_uop_rs2ex[0].rs1_data_valid,
    //                 mxu_uop_rs2ex[0].uop_funct6,
    //                 mxu_uop_rs2ex[0].uop_funct3,
    //                 mxu_uop_rs2ex[0].vm);
    //     end
    // end

endmodule