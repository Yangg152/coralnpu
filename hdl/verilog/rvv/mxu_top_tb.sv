// file: mxu_top_tb.sv
`ifndef HDL_VERILOG_RVV_DESIGN_RVV_SVH
`include "rvv_backend.svh"
`endif

module mxu_top_tb;

  // ============================================================
  // Parameters
  // ============================================================
  localparam TK = 16;
  localparam NUM_W_BEATS  = TK;
  localparam NUM_A_BEATS  = TK;
  localparam NUM_STORE    = 64;

  // ============================================================
  // Clock & Reset
  // ============================================================
  logic clk, rst_n;
  initial clk = 0;
  always #5 clk = ~clk;

  initial begin
    rst_n = 0;
    repeat(10) @(posedge clk);
    rst_n = 1;
  end

  // ============================================================
  // DUT signals
  // ============================================================
  logic [`ISSUE_LANE-1:0]           insts_valid;
  RVVCmd [`ISSUE_LANE-1:0]          insts;
  logic [`ISSUE_LANE-1:0]           insts_ready;
  logic [$clog2(`CQ_DEPTH):0]       remaining_count;

  logic [`NUM_LSU-1:0]              uop_lsu_valid_rvv2lsu;
  UOP_RVV2LSU_t [`NUM_LSU-1:0]     uop_lsu_rvv2lsu;
  logic [`NUM_LSU-1:0]              uop_lsu_ready_lsu2rvv;

  logic [`NUM_LSU-1:0]              uop_lsu_valid_lsu2rvv;
  UOP_LSU2RVV_t [`NUM_LSU-1:0]     uop_lsu_lsu2rvv;
  logic [`NUM_LSU-1:0]              uop_lsu_ready_rvv2lsu;

  logic [`NUM_RT_UOP-1:0]           rt_xrf_valid;
  RT2XRF_t [`NUM_RT_UOP-1:0]       rt_xrf;
  logic [`NUM_RT_UOP-1:0]           rt_xrf_ready;

  logic                             wr_vxsat_valid;
  logic [`VCSR_VXSAT_WIDTH-1:0]     wr_vxsat;
  logic                             wr_vxsat_ready;

  logic                             trap_valid;
  logic                             trap_ready;

  logic                             vcsr_valid;
  RVVConfigState                    vector_csr;
  logic                             vcsr_ready;

  logic [`NUM_RT_UOP-1:0]           rd_valid_rob2rt_o;
  logic                             rvv_idle;
  ROB2RT_t [`NUM_RT_UOP-1:0]        rd_rob2rt_o;

  // ============================================================
  // DUT
  // ============================================================
  rvv_backend u_dut (
    .clk                    (clk),
    .rst_n                  (rst_n),
    .insts_valid_rvs2cq     (insts_valid),
    .insts_rvs2cq           (insts),
    .insts_ready_cq2rvs     (insts_ready),
    .remaining_count_cq2rvs (remaining_count),
    .uop_lsu_valid_rvv2lsu  (uop_lsu_valid_rvv2lsu),
    .uop_lsu_rvv2lsu        (uop_lsu_rvv2lsu),
    .uop_lsu_ready_lsu2rvv  (uop_lsu_ready_lsu2rvv),
    .uop_lsu_valid_lsu2rvv  (uop_lsu_valid_lsu2rvv),
    .uop_lsu_lsu2rvv        (uop_lsu_lsu2rvv),
    .uop_lsu_ready_rvv2lsu  (uop_lsu_ready_rvv2lsu),
    .rt_xrf_rvv2rvs         (rt_xrf),
    .rt_xrf_valid_rvv2rvs   (rt_xrf_valid),
    .rt_xrf_ready_rvs2rvv   (rt_xrf_ready),
    .wr_vxsat_valid         (wr_vxsat_valid),
    .wr_vxsat               (wr_vxsat),
    .wr_vxsat_ready         (wr_vxsat_ready),
    .trap_valid_rvs2rvv     (trap_valid),
    .trap_ready_rvv2rvs     (trap_ready),
    .vcsr_valid             (vcsr_valid),
    .vector_csr             (vector_csr),
    .vcsr_ready             (vcsr_ready),
  `ifdef TB_SUPPORT
    .rd_valid_rob2rt_o      (rd_valid_rob2rt_o),
  `endif
    .rvv_idle               (rvv_idle),
    .rd_rob2rt_o            (rd_rob2rt_o)
  );

  // ============================================================
  // Tie-off
  // ============================================================
  assign uop_lsu_ready_lsu2rvv = '0;
  assign uop_lsu_valid_lsu2rvv = '0;
  assign uop_lsu_lsu2rvv       = '0;
  assign rt_xrf_ready          = '1;
  assign vcsr_ready            = 1'b1;
  assign wr_vxsat_ready        = 1'b1;
  assign trap_valid            = 1'b0;

  // ============================================================
  // Test data
  // ============================================================
  logic signed [7:0]  weight_matrix [0:TK-1][0:15];
  logic signed [7:0]  act_matrix    [0:15][0:TK-1];
  logic signed [31:0] golden        [0:15][0:15];
  logic [127:0]       golden_packed [0:63];
  logic [127:0]       weight_vectors [0:TK-1];
  logic [127:0]       act_vectors    [0:NUM_A_BEATS-1];

  initial begin
    for (int k = 0; k < TK; k++)
      for (int n = 0; n < 16; n++)
        weight_matrix[k][n] = $random % 64;

    for (int m = 0; m < 16; m++)
      for (int k = 0; k < TK; k++)
        act_matrix[m][k] = $random % 64;

    for (int m = 0; m < 16; m++)
      for (int n = 0; n < 16; n++) begin
        golden[m][n] = 0;
        for (int k = 0; k < TK; k++)
          golden[m][n] += act_matrix[m][k] * weight_matrix[k][n];
      end

    for (int m = 0; m < 16; m++)
      for (int g = 0; g < 4; g++)
        golden_packed[m*4+g] = {golden[m][g*4+3], golden[m][g*4+2],
                                golden[m][g*4+1], golden[m][g*4+0]};

    for (int k = 0; k < TK; k++) begin
      weight_vectors[k] = '0;
      for (int n = 0; n < 16; n++)
        weight_vectors[k][n*8 +: 8] = weight_matrix[k][n];
    end

    for (int m = 0; m < 16; m++) begin
      act_vectors[m] = '0;
      for (int j = 0; j < 16; j++)
        act_vectors[m][j*8 +: 8] = act_matrix[m][j];
    end
  end

  // ============================================================
  // VRF backdoor write - static signals for force
  // ============================================================
  logic [`NUM_VRF-1:0][`VLENB-1:0] vrf_force_wen;
  logic [`NUM_VRF-1:0][`VLEN-1:0]  vrf_force_wdata;

  task automatic init_vrf_reg(input int reg_idx, input logic [`VLEN-1:0] data);
    vrf_force_wen   = '0;
    vrf_force_wdata = '0;
    vrf_force_wen[reg_idx]   = {`VLENB{1'b1}};
    vrf_force_wdata[reg_idx] = data;

    @(negedge clk);
    force u_dut.u_vrf.vrf_wr_wen_full  = vrf_force_wen;
    force u_dut.u_vrf.vrf_wr_data_full = vrf_force_wdata;
    @(posedge clk);
    @(negedge clk);
    release u_dut.u_vrf.vrf_wr_wen_full;
    release u_dut.u_vrf.vrf_wr_data_full;
  endtask

  task automatic init_all_vrf();
    $display("[TB] Initializing VRF...");
    // v1~v16: weight rows 0~15
    for (int k = 0; k < 16; k++)
      init_vrf_reg(k + 1, weight_vectors[k]);
    // v17~v31: activation rows 0~14
    for (int m = 0; m < 15; m++)
      init_vrf_reg(m + 17, act_vectors[m]);
    $display("[TB] VRF init done.");
  endtask

  // ============================================================
  // Verify VRF contents after init (using TB_SUPPORT vrf_data)
  // ============================================================
  task automatic verify_vrf_init();
  `ifdef TB_SUPPORT
    logic [`VLEN-1:0] readback;
    int errs = 0;
    for (int k = 0; k < 16; k++) begin
      readback = u_dut.u_vrf.vrf_data[k+1];
      if (readback !== weight_vectors[k]) begin
        $display("[TB] VRF INIT ERR: v%0d got=%h exp=%h", k+1, readback, weight_vectors[k]);
        errs++;
      end
    end
    for (int m = 0; m < 15; m++) begin
      readback = u_dut.u_vrf.vrf_data[m+17];
      if (readback !== act_vectors[m]) begin
        $display("[TB] VRF INIT ERR: v%0d got=%h exp=%h", m+17, readback, act_vectors[m]);
        errs++;
      end
    end
    if (errs == 0)
      $display("[TB] VRF init verification PASSED.");
    else
      $display("[TB] VRF init verification FAILED with %0d errors.", errs);
  `endif
  endtask

  // ============================================================
  // RVVCmd helpers
  // ============================================================
  function automatic RVVConfigState mxu_arch_state();
    RVVConfigState s;
    s        = '0;
    s.vill   = 1'b0;
    s.vl     = 'd16;
    s.vstart = '0;
    s.ma     = 1'b1;
    s.ta     = 1'b1;
    s.xrm    = RNU;
    s.xsat   = 1'b0;
    s.sew    = SEW8;
    s.lmul   = LMUL1;
    s.lmul_orig = LMUL1;
    return s;
  endfunction

  function automatic logic [24:0] encode_bits(
    input logic [5:0] funct6,
    input logic       vm,
    input logic [4:0] vs2,
    input logic [4:0] rs1_field,
    input logic [4:0] vd
  );
    logic [24:0] b;
    b[24:19] = funct6;
    b[18]    = vm;
    b[17:13] = vs2;
    b[12:8]  = rs1_field;
    b[7:5]   = OPMXU;
    b[4:0]   = vd;
    return b;
  endfunction

  function automatic RVVCmd make_cmd(
    input logic [5:0]  funct6,
    input logic        vm,
    input logic [4:0]  vs2,
    input logic [4:0]  rs1_field,
    input logic [4:0]  vd,
    input logic [31:0] rs1_data
  );
    RVVCmd cmd;
    cmd = '0;
  `ifdef TB_SUPPORT
    cmd.inst_pc = 32'hBEEF_0000;
  `endif
    cmd.opcode     = RVV;
    cmd.bits       = encode_bits(funct6, vm, vs2, rs1_field, vd);
    cmd.rs1        = rs1_data;
    cmd.arch_state = mxu_arch_state();
    return cmd;
  endfunction

  // ============================================================
  // Send single command on lane 0
  // ============================================================
  task automatic send_cmd(input RVVCmd cmd);
    insts_valid    <= '0;
    insts_valid[0] <= 1'b1;
    insts[0]       <= cmd;
    @(posedge clk);
    while (!insts_ready[0]) @(posedge clk);
    insts_valid <= '0;
  endtask

  task automatic wait_idle(input int timeout);
    int cnt = 0;
    while (!rvv_idle && cnt < timeout) begin
      @(posedge clk);
      cnt++;
    end
    if (cnt >= timeout)
      $display("[TB] WARN: wait_idle timeout %0d cycles @ T=%0t", timeout, $time);
  endtask

  // ============================================================
  // Main test
  // ============================================================
  int retire_count;
  int store_count;
  int mismatch_count;

  initial begin
    $fsdbDumpfile("mxu_top_tb.fsdb");
    $fsdbDumpvars(0, mxu_top_tb, "+all");

    insts_valid    = '0;
    insts          = '0;
    retire_count   = 0;
    store_count    = 0;
    mismatch_count = 0;

    @(posedge rst_n);
    repeat(5) @(posedge clk);

    // ---- Init VRF ----
    init_all_vrf();
    repeat(3) @(posedge clk);
    verify_vrf_init();
    repeat(5) @(posedge clk);

    $display("[TB] ===== MXU Integration Test: Tk=%0d =====", TK);

    // ---- MCFG ----
    $display("[TB] MCFG Tk=%0d", TK);
    send_cmd(make_cmd(MXU_MCFG, 1'b1, 5'd0, 5'd0, 5'd0, 32'd16));
    wait_idle(200);

    // ---- MZERO ----
    $display("[TB] MZERO");
    send_cmd(make_cmd(MXU_MZERO, 1'b1, 5'd0, 5'd0, 5'd0, 32'd0));
    wait_idle(200);

    // ---- MLOAD_W: vs2 = v1~v16 ----
    $display("[TB] MLOAD_W x%0d", NUM_W_BEATS);
    for (int k = 0; k < NUM_W_BEATS; k++) begin
      logic vm_bit;
      vm_bit = (k == NUM_W_BEATS - 1) ? 1'b0 : 1'b1;
      send_cmd(make_cmd(MXU_MLOAD_W, vm_bit, 5'(k+1), 5'd0, 5'd0, 32'd0));
    end
    wait_idle(500);

    // ---- Write act[15] into v1 (reuse after weights consumed) ----
    init_vrf_reg(1, act_vectors[15]);
    repeat(3) @(posedge clk);

    // ---- MLOAD_A: vs2 = v17~v31 for rows 0~14, v1 for row 15 ----
    $display("[TB] MLOAD_A x%0d", NUM_A_BEATS);
    for (int beat = 0; beat < NUM_A_BEATS; beat++) begin
      logic vm_bit;
      logic [4:0] vs2_idx;
      vm_bit = (beat == NUM_A_BEATS - 1) ? 1'b0 : 1'b1;
      if (beat < 15)
        vs2_idx = 5'(beat + 17);
      else
        vs2_idx = 5'd1;
      send_cmd(make_cmd(MXU_MLOAD_A, vm_bit, vs2_idx, 5'd0, 5'd0, 32'd0));
    end
    wait_idle(500);

    // ---- MMA ----
    $display("[TB] MMA");
    send_cmd(make_cmd(MXU_MMA, 1'b1, 5'd0, 5'd0, 5'd0, 32'd0));
    wait_idle(500);

    // ---- MSTORE x64 ----
    $display("[TB] MSTORE x%0d", NUM_STORE);
    for (int s = 0; s < NUM_STORE; s++) begin
      send_cmd(make_cmd(MXU_MSTORE, 1'b1, 5'd0, 5'd0, 5'd3, 32'd0));
    end
    wait_idle(5000);

    repeat(500) @(posedge clk);

    $display("[TB] ===== Results =====");
    $display("[TB] Retirements: %0d", retire_count);
    $display("[TB] MSTORE results: %0d / %0d", store_count, NUM_STORE);
    $display("[TB] Mismatches: %0d", mismatch_count);

    if (mismatch_count == 0 && store_count == NUM_STORE)
      $display("[TB] >>> PASS <<<");
    else
      $display("[TB] >>> FAIL <<<");

    $finish;
  end

  // ============================================================
  // Retire monitor
  // ============================================================
  int expected_store_idx;
  initial expected_store_idx = 0;

  always @(posedge clk) begin
    if (rst_n) begin
      for (int i = 0; i < `NUM_RT_UOP; i++) begin
        if (rd_valid_rob2rt_o[i]) begin
          retire_count++;
          if (rd_rob2rt_o[i].w_valid && rd_rob2rt_o[i].w_type == VRF) begin
            logic [127:0] actual;
            actual = rd_rob2rt_o[i].w_data[127:0];
            if (expected_store_idx < NUM_STORE) begin
              if (actual !== golden_packed[expected_store_idx]) begin
                $display("[TB] MISMATCH beat[%0d]: got=%h exp=%h",
                         expected_store_idx, actual, golden_packed[expected_store_idx]);
                mismatch_count++;
              end else if (expected_store_idx < 4 || expected_store_idx >= NUM_STORE-2) begin
                $display("[TB] OK beat[%0d]: %h", expected_store_idx, actual);
              end
              expected_store_idx++;
            end
            store_count++;
          end
        end
      end
    end
  end

  // ============================================================
  // Debug: CQ push monitor
  // ============================================================
  always @(posedge clk) begin
    if (rst_n) begin
      for (int i = 0; i < `ISSUE_LANE; i++)
        if (insts_valid[i] && insts_ready[i])
          $display("[TB] T=%0t CQ[%0d] funct6=%b funct3=%b vm=%b vs2=%0d rs1=0x%h",
                   $time, i,
                   insts[i].bits[24:19], insts[i].bits[7:5],
                   insts[i].bits[18], insts[i].bits[17:13],
                   insts[i].rs1);
    end
  end

  // ============================================================
  // Watchdog
  // ============================================================
  initial begin
    #2_000_000;
    $display("[TB] TIMEOUT");
    $finish;
  end

endmodule