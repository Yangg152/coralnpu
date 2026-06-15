`timescale 1ns/1ps

module mxu_tb;

parameter CLK_PERIOD = 10;
parameter MAX_TK     = 256;

reg         clk, rst_n;
reg  [2:0]  op_type;
reg         op_valid;
wire        op_ready;
wire        op_done;
reg  [5:0]  uop_index;
reg         uop_last;
reg  [7:0]  cfg_Tk;
reg         cfg_signed;
reg  [127:0] weight_vec;
reg          weight_valid;
wire         weight_ready;
reg  [127:0] act_vec;
reg          act_valid;
wire         act_ready;
wire [127:0] result_data;
wire         result_valid;
reg          result_ready;
wire         mfence_done;

rvv_backend_mxu_unit u_dut (
    .clk          (clk),
    .rst_n        (rst_n),
    .op_type      (op_type),
    .op_valid     (op_valid),
    .op_ready     (op_ready),
    .op_done      (op_done),
    .uop_index    (uop_index),
    .uop_last     (uop_last),
    .cfg_Tk       (cfg_Tk),
    .cfg_signed   (cfg_signed),
    .weight_vec   (weight_vec),
    .weight_valid (weight_valid),
    .weight_ready (weight_ready),
    .act_vec      (act_vec),
    .act_valid    (act_valid),
    .act_ready    (act_ready),
    .result_data  (result_data),
    .result_valid (result_valid),
    .result_ready (result_ready),
    .mfence_done  (mfence_done)
);

initial clk = 0;
always #(CLK_PERIOD/2) clk = ~clk;

initial begin
    $fsdbDumpfile("mxu_wave.fsdb");
    $fsdbDumpvars(0, mxu_tb, "+all");
    $fsdbDumpMDA;
end

integer watchdog;
initial begin
    watchdog = 0;
    forever begin
        @(posedge clk);
        watchdog = watchdog + 1;
        if (watchdog > 500000) begin
            $display("[TIMEOUT] Simulation exceeded watchdog limit!");
            $finish;
        end
    end
end

// always @(posedge clk) begin
//     if (u_dut.pe_en) begin
//         $display("[PE] k=%0d w_rd=%h a_rd=%h",
//                  u_dut.k_cnt_p1,
//                  u_dut.wbuf_rdata,
//                  u_dut.abuf_rdata);
//     end
// end

// ---------------------------------------------------------------
// Test data storage
// ---------------------------------------------------------------
reg [127:0] weight_mem [0:MAX_TK-1];
reg [127:0] act_mem    [0:16*16-1];
reg [127:0] golden_mem [0:63];

integer total_pass, total_fail;

// ---------------------------------------------------------------
// Helper tasks
// All signal drives use #1 after @(posedge clk) to avoid
// same-edge delta-cycle race with the DUT.
// ---------------------------------------------------------------
task automatic do_reset;
begin
    rst_n = 0;
    op_valid = 0; op_type = 0; uop_last = 0;
    weight_valid = 0; act_valid = 0;
    result_ready = 1; cfg_signed = 1;
    uop_index = 0; cfg_Tk = 0;
    weight_vec = 0; act_vec = 0;
    repeat(5) @(posedge clk);
    rst_n = 1;
    repeat(2) @(posedge clk);
end
endtask

task automatic do_mcfg(input [7:0] tk_val);
begin
    @(posedge clk); #1;
    while (!op_ready) begin @(posedge clk); #1; end
    op_type  = 3'd0;
    cfg_Tk   = tk_val;
    op_valid = 1;
    uop_last = 1;
    @(posedge clk); #1;
    op_valid = 0;
    uop_last = 0;
    while (!op_done) begin @(posedge clk); #1; end
    @(posedge clk); #1;
    $display("  [do_mcfg] Tk=%0d done", tk_val);
end
endtask

task automatic do_mzero;
begin
    @(posedge clk); #1;
    while (!op_ready) begin @(posedge clk); #1; end
    op_type  = 3'd3;
    op_valid = 1;
    uop_last = 1;
    @(posedge clk); #1;
    op_valid = 0;
    uop_last = 0;
    while (!op_done) begin @(posedge clk); #1; end
    @(posedge clk); #1;
    $display("  [do_mzero] done");
end
endtask

// ---------------------------------------------------------------
// do_mload_w
//
// RTL protocol:
//   Beat 0: S_IDLE sees op_valid && OP_MLOAD_W
//           -> combinational wbuf write at addr=load_cnt with weight_vec
//           -> load_cnt++, state -> S_LOAD_W, weight_ready=1
//   Beat 1+: S_LOAD_W sees weight_valid
//           -> combinational wbuf write at addr=load_cnt with weight_vec
//           -> load_cnt++
//           -> if uop_last: op_done, back to S_READY
//
// TB must have weight_vec stable BEFORE the posedge where DUT samples.
// Using @(posedge clk); #1 ensures our drives settle before next posedge.
// ---------------------------------------------------------------
task automatic do_mload_w(input integer num_beats);
    integer i;
begin
    $display("  [do_mload_w] num_beats=%0d", num_beats);

    // Beat 0: present op_valid + weight_vec together
    @(posedge clk); #1;
    while (!op_ready) begin @(posedge clk); #1; end
    op_type      = 3'd1;
    op_valid     = 1;
    weight_vec   = weight_mem[0];
    weight_valid = 1;
    uop_last     = (num_beats == 1) ? 1 : 0;

    // Beats 1..N-1
    for (i = 1; i < num_beats; i = i + 1) begin
        @(posedge clk); #1;
        op_valid = 0;
        while (!weight_ready) begin @(posedge clk); #1; end
        weight_vec   = weight_mem[i];
        weight_valid = 1;
        uop_last     = (i == num_beats - 1) ? 1 : 0;
    end

    @(posedge clk); #1;
    op_valid     = 0;
    weight_valid = 0;
    uop_last     = 0;
    while (!op_done) begin @(posedge clk); #1; end
    @(posedge clk); #1;
    $display("  [do_mload_w] done");
end
endtask

// ---------------------------------------------------------------
// do_mload_a
//
// RTL protocol:
//   Beat 0: S_IDLE sees op_valid && OP_MLOAD_A
//           -> latch act_vec, go to S_LOAD_A_EX (16-cycle expand)
//   Beat 1+: S_LOAD_A sees act_valid
//           -> latch act_vec, go to S_LOAD_A_EX
// ---------------------------------------------------------------
task automatic do_mload_a(input integer num_beats);
    integer i;
begin
    $display("  [do_mload_a] num_beats=%0d", num_beats);

    // Beat 0
    @(posedge clk); #1;
    while (!op_ready) begin @(posedge clk); #1; end
    op_type   = 3'd2;
    op_valid  = 1;
    act_vec   = act_mem[0];
    act_valid = 1;
    uop_last  = (num_beats == 1) ? 1 : 0;

    // Beats 1..N-1
    for (i = 1; i < num_beats; i = i + 1) begin
        @(posedge clk); #1;
        op_valid = 0;
        while (!act_ready) begin @(posedge clk); #1; end
        act_vec   = act_mem[i];
        act_valid = 1;
        uop_last  = (i == num_beats - 1) ? 1 : 0;
    end

    @(posedge clk); #1;
    op_valid  = 0;
    act_valid = 0;
    uop_last  = 0;
    while (!op_done) begin @(posedge clk); #1; end
    @(posedge clk); #1;
    $display("  [do_mload_a] done");
end
endtask

task automatic do_mma;
begin
    @(posedge clk); #1;
    while (!op_ready) begin @(posedge clk); #1; end
    op_type  = 3'd4;
    op_valid = 1;
    uop_last = 1;
    @(posedge clk); #1;
    op_valid = 0;
    uop_last = 0;
    while (!op_done) begin @(posedge clk); #1; end
    @(posedge clk); #1;
    $display("  [do_mma] done");
end
endtask

task automatic do_mstore_and_check(input integer test_id);
    integer i, pass_cnt, fail_cnt;
    reg [127:0] expected;
begin
    pass_cnt = 0;
    fail_cnt = 0;
    for (i = 0; i < 64; i = i + 1) begin
        @(posedge clk); #1;
        while (!op_ready) begin @(posedge clk); #1; end
        op_type  = 3'd5;
        op_valid = 1;
        uop_last = 1;
        @(posedge clk); #1;
        op_valid = 0;
        uop_last = 0;
        while (!op_done) begin @(posedge clk); #1; end

        expected = golden_mem[i];
        if (result_data === expected) begin
            pass_cnt = pass_cnt + 1;
        end else begin
            fail_cnt = fail_cnt + 1;
            if (fail_cnt <= 5)
                $display("[FAIL] Test%0d Beat%0d: got=%h exp=%h",
                         test_id, i, result_data, expected);
        end
    end
    total_pass = total_pass + pass_cnt;
    total_fail = total_fail + fail_cnt;
    $display("[TEST%0d] %0d/64 passed, %0d failed", test_id, pass_cnt, fail_cnt);
end
endtask

function integer calc_a_beats;
    input integer Tk;
    integer chunks;
begin
    chunks = (Tk + 15) / 16;
    calc_a_beats = 16 * chunks;
end
endfunction

// ---------------------------------------------------------------
// Main
// ---------------------------------------------------------------
initial begin
    total_pass = 0;
    total_fail = 0;
    do_reset;

    $display("======== TEST1: Tk=16 ========");
    $readmemh("cases/tc1_tk16_weight.hex", weight_mem);
    $readmemh("cases/tc1_tk16_act.hex",    act_mem);
    $readmemh("cases/tc1_tk16_golden.hex", golden_mem);
    do_mcfg(8'd16);
    do_mzero;
    do_mload_w(16);
    do_mload_a(calc_a_beats(16));
    do_mma;
    do_mstore_and_check(1);

    $display("======== TEST2: Tk=64 ========");
    $readmemh("cases/tc2_tk64_weight.hex", weight_mem);
    $readmemh("cases/tc2_tk64_act.hex",    act_mem);
    $readmemh("cases/tc2_tk64_golden.hex", golden_mem);
    do_mcfg(8'd64);
    do_mzero;
    do_mload_w(64);
    do_mload_a(calc_a_beats(64));
    do_mma;
    do_mstore_and_check(2);

    $display("======== TEST3: Tk=256 ========");
    $readmemh("cases/tc3_tk256_weight.hex", weight_mem);
    $readmemh("cases/tc3_tk256_act.hex",    act_mem);
    $readmemh("cases/tc3_tk256_golden.hex", golden_mem);
    do_mcfg(8'd0);
    do_mzero;
    do_mload_w(256);
    do_mload_a(calc_a_beats(256));
    do_mma;
    do_mstore_and_check(3);

    $display("======== TEST4: unsigned Tk=16 ========");
    $readmemh("cases/tc4_unsigned_weight.hex", weight_mem);
    $readmemh("cases/tc4_unsigned_act.hex",    act_mem);
    $readmemh("cases/tc4_unsigned_golden.hex", golden_mem);
    do_mcfg(8'd16);
    do_mzero;
    do_mload_w(16);
    do_mload_a(calc_a_beats(16));
    do_mma;
    do_mstore_and_check(4);

    $display("");
    $display("============ SUMMARY ============");
    $display("  PASS: %0d", total_pass);
    $display("  FAIL: %0d", total_fail);
    if (total_fail == 0)
        $display("  *** ALL TESTS PASSED ***");
    else
        $display("  *** SOME TESTS FAILED ***");
    $display("=================================");

    #100;
    $finish;
end

endmodule