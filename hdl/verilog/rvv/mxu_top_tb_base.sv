`timescale 1ns / 1ps

`define TB_SUPPORT

`ifndef HDL_VERILOG_RVV_DESIGN_RVV_SVH
`include "rvv_backend.svh"
`endif

module mxu_top_tb;

    localparam CLK_HALF = 5;
    reg clk = 0;
    always #CLK_HALF clk = ~clk;

    logic                                     rst_n;
    logic             [`ISSUE_LANE-1:0]       insts_valid_rvs2cq;
    RVVCmd            [`ISSUE_LANE-1:0]       insts_rvs2cq;
    logic             [`ISSUE_LANE-1:0]       insts_ready_cq2rvs;
    logic             [$clog2(`CQ_DEPTH):0]   remaining_count_cq2rvs;
    logic             [`NUM_LSU-1:0]          uop_lsu_valid_rvv2lsu;
    UOP_RVV2LSU_t     [`NUM_LSU-1:0]          uop_lsu_rvv2lsu;
    logic             [`NUM_LSU-1:0]          uop_lsu_ready_lsu2rvv;
    logic             [`NUM_LSU-1:0]          uop_lsu_valid_lsu2rvv;
    UOP_LSU2RVV_t     [`NUM_LSU-1:0]          uop_lsu_lsu2rvv;
    logic             [`NUM_LSU-1:0]          uop_lsu_ready_rvv2lsu;
    logic             [`NUM_RT_UOP-1:0]       rt_xrf_valid_rvv2rvs;
    RT2XRF_t          [`NUM_RT_UOP-1:0]       rt_xrf_rvv2rvs;
    logic             [`NUM_RT_UOP-1:0]       rt_xrf_ready_rvs2rvv;
    logic                                     wr_vxsat_valid;
    logic             [`VCSR_VXSAT_WIDTH-1:0] wr_vxsat;
    logic                                     wr_vxsat_ready;
    logic                                     trap_valid_rvs2rvv;
    logic                                     trap_ready_rvv2rvs;
    logic                                     vcsr_valid;
    RVVConfigState                            vector_csr;
    logic                                     vcsr_ready;
    logic             [`NUM_RT_UOP-1:0]       rd_valid_rob2rt_o;
    logic                                     rvv_idle;
    ROB2RT_t          [`NUM_RT_UOP-1:0]       rd_rob2rt_o;

    // =========================================================
    // DUT
    // =========================================================
    rvv_backend u_dut (
        .clk                    (clk),
        .rst_n                  (rst_n),
        .insts_valid_rvs2cq     (insts_valid_rvs2cq),
        .insts_rvs2cq           (insts_rvs2cq),
        .insts_ready_cq2rvs     (insts_ready_cq2rvs),
        .remaining_count_cq2rvs (remaining_count_cq2rvs),
        .uop_lsu_valid_rvv2lsu  (uop_lsu_valid_rvv2lsu),
        .uop_lsu_rvv2lsu        (uop_lsu_rvv2lsu),
        .uop_lsu_ready_lsu2rvv  (uop_lsu_ready_lsu2rvv),
        .uop_lsu_valid_lsu2rvv  (uop_lsu_valid_lsu2rvv),
        .uop_lsu_lsu2rvv        (uop_lsu_lsu2rvv),
        .uop_lsu_ready_rvv2lsu  (uop_lsu_ready_rvv2lsu),
        .rt_xrf_valid_rvv2rvs   (rt_xrf_valid_rvv2rvs),
        .rt_xrf_rvv2rvs         (rt_xrf_rvv2rvs),
        .rt_xrf_ready_rvs2rvv   (rt_xrf_ready_rvs2rvv),
        .wr_vxsat_valid         (wr_vxsat_valid),
        .wr_vxsat               (wr_vxsat),
        .wr_vxsat_ready         (wr_vxsat_ready),
        .trap_valid_rvs2rvv     (trap_valid_rvs2rvv),
        .trap_ready_rvv2rvs     (trap_ready_rvv2rvs),
        .vcsr_valid             (vcsr_valid),
        .vector_csr             (vector_csr),
        .vcsr_ready             (vcsr_ready),
        .rd_valid_rob2rt_o      (rd_valid_rob2rt_o),
        .rvv_idle               (rvv_idle),
        .rd_rob2rt_o            (rd_rob2rt_o)
    );

    // =========================================================
    // 外部接口默认值
    // =========================================================
    assign uop_lsu_ready_lsu2rvv = '0;
    assign uop_lsu_valid_lsu2rvv = '0;
    assign uop_lsu_lsu2rvv       = '0;
    assign rt_xrf_ready_rvs2rvv  = '1;
    assign wr_vxsat_ready        = 1'b1;
    assign vcsr_ready            = 1'b1;
    assign trap_valid_rvs2rvv    = 1'b0;

    // =========================================================
    // 计数器
    // =========================================================
    int watchdog;
    localparam WATCHDOG_MAX = 100000;
    int retire_cnt, mstore_cnt;
    int total_pass, total_fail;

    // =========================================================
    // MSTORE 数据捕获
    // =========================================================
    localparam MAX_CAPTURE = 1024;
    logic [127:0] captured_data [0:MAX_CAPTURE-1];
    logic [4:0]   captured_vd   [0:MAX_CAPTURE-1];
    int           capture_idx;

    always @(posedge clk) begin
        for (int i = 0; i < `NUM_RT_UOP; i++) begin
            if (rd_valid_rob2rt_o[i]) begin
                retire_cnt++;
                if (rd_rob2rt_o[i].w_valid) begin
                    mstore_cnt++;
                    if (capture_idx < MAX_CAPTURE) begin
                        captured_data[capture_idx] = rd_rob2rt_o[i].w_data[127:0];
                        captured_vd[capture_idx]   = rd_rob2rt_o[i].w_index;
                        capture_idx++;
                    end
                end
            end
        end
    end

    // =========================================================
    // VRF backdoor 写入
    // =========================================================
    logic                              tb_vrf_wr_en;
    logic [`REGFILE_INDEX_WIDTH-1:0]   tb_vrf_wr_idx;
    logic [`VLEN-1:0]                  tb_vrf_wr_data;
    logic [`VLENB-1:0]                 tb_vrf_wr_strobe;

    always @(*) begin
        if (tb_vrf_wr_en) begin
            force u_dut.u_vrf.rt2vrf_wr_valid[0]          = 1'b1;
            force u_dut.u_vrf.rt2vrf_wr_data[0].rt_index  = tb_vrf_wr_idx;
            force u_dut.u_vrf.rt2vrf_wr_data[0].rt_data   = tb_vrf_wr_data;
            force u_dut.u_vrf.rt2vrf_wr_data[0].rt_strobe = tb_vrf_wr_strobe;
        end else begin
            release u_dut.u_vrf.rt2vrf_wr_valid[0];
            release u_dut.u_vrf.rt2vrf_wr_data[0].rt_index;
            release u_dut.u_vrf.rt2vrf_wr_data[0].rt_data;
            release u_dut.u_vrf.rt2vrf_wr_data[0].rt_strobe;
        end
    end

    task automatic vrf_write(input int vreg_idx, input logic [`VLEN-1:0] data);
        @(negedge clk);
        tb_vrf_wr_en     = 1'b1;
        tb_vrf_wr_idx    = vreg_idx[`REGFILE_INDEX_WIDTH-1:0];
        tb_vrf_wr_data   = data;
        tb_vrf_wr_strobe = {`VLENB{1'b1}};
        @(posedge clk);
        @(negedge clk);
        tb_vrf_wr_en     = 1'b0;
    endtask

    task automatic vrf_write_bytes(input int vreg_idx, input byte unsigned bdata[16]);
        logic [`VLEN-1:0] packed_data;
        packed_data = '0;
        for (int i = 0; i < 16; i++)
            packed_data[i*8 +: 8] = bdata[i];
        vrf_write(vreg_idx, packed_data);
    endtask

    // =========================================================
    // 指令构建
    // =========================================================
    function automatic RVVCmd build_mxu_cmd(
        input [5:0]  funct6, input vm,
        input [4:0]  vs2, input [4:0] vs1, input [4:0] vd,
        input [31:0] rs1_val
    );
        RVVCmd cmd;
        cmd.opcode = RVV;
        cmd.bits   = {funct6, vm, vs2, vs1, OPMXU, vd};
        cmd.rs1    = rs1_val;
        cmd.inst_pc = 32'hDEAD_0000;
        cmd.arch_state.vill       = 1'b0;
        cmd.arch_state.vl         = 'd16;
        cmd.arch_state.vstart     = '0;
        cmd.arch_state.ma         = 1'b1;
        cmd.arch_state.ta         = 1'b1;
        cmd.arch_state.xrm        = RNU;
        cmd.arch_state.xsat       = 1'b0;
        cmd.arch_state.sew        = SEW8;
        cmd.arch_state.lmul       = LMUL1;
        cmd.arch_state.lmul_orig  = LMUL1;
        return cmd;
    endfunction

    function automatic RVVCmd build_mstore_cmd(input [4:0] vd);
        RVVCmd cmd;
        cmd.opcode = RVV;
        cmd.bits   = {MXU_MSTORE, 1'b1, 5'd0, 5'd0, OPIVI, vd};
        cmd.rs1    = 32'd0;
        cmd.inst_pc = 32'hDEAD_0000;
        cmd.arch_state.vill       = 1'b0;
        cmd.arch_state.vl         = 'd16;
        cmd.arch_state.vstart     = '0;
        cmd.arch_state.ma         = 1'b1;
        cmd.arch_state.ta         = 1'b1;
        cmd.arch_state.xrm        = RNU;
        cmd.arch_state.xsat       = 1'b0;
        cmd.arch_state.sew        = SEW8;
        cmd.arch_state.lmul       = LMUL1;
        cmd.arch_state.lmul_orig  = LMUL1;
        return cmd;
    endfunction

    // =========================================================
    // 基础工具 task
    // =========================================================
    task automatic send_cmd(input RVVCmd cmd);
        @(negedge clk);
        insts_valid_rvs2cq    = '0;
        insts_valid_rvs2cq[0] = 1'b1;
        insts_rvs2cq[0]       = cmd;
        watchdog = 0;
        forever begin
            @(posedge clk);
            if (insts_ready_cq2rvs[0]) break;
            watchdog++;
            if (watchdog > WATCHDOG_MAX) begin
                $display("[HANG] send_cmd stuck t=%0t", $time); $finish;
            end
        end
        @(negedge clk);
        insts_valid_rvs2cq = '0;
    endtask

    task automatic wait_idle();
        watchdog = 0;
        forever begin
            @(posedge clk);
            if (rvv_idle) break;
            watchdog++;
            if (watchdog > WATCHDOG_MAX) begin
                $display("[HANG] wait_idle stuck t=%0t", $time); $finish;
            end
        end
    endtask

    task automatic reset_cnt();
        retire_cnt  = 0;
        mstore_cnt  = 0;
        capture_idx = 0;
    endtask

    task automatic check(input string name, input bit cond);
        if (cond) begin $display("  [PASS] %s", name); total_pass++; end
        else      begin $display("  [FAIL] %s", name); total_fail++; end
    endtask

    // =========================================================
    // 指令原语
    //
    // RTL 变化要点:
    //   - cfg_Tk 必须是 16 的倍数, num_a_chunks = cfg_Tk >> 4
    //   - MLOAD_W: 多拍, 每拍一条指令, uop_last 标记最后一拍
    //   - MLOAD_A: 一条指令触发, 后续拍通过 act_valid/act_ready
    //              握手从 VRF 读取, 共 16 * num_chunks 拍
    //              TB 端只需发一条 MLOAD_A 指令
    //   - MMA: 内部 k_cnt 自动迭代 cfg_Tk 拍
    // =========================================================
    task automatic do_mcfg(input int Tk, input bit is_signed = 1'b1);
        send_cmd(build_mxu_cmd(MXU_MCFG, 1'b1, 5'd0, 5'd0, 5'd0,
                 (Tk & 32'hFF) | (is_signed ? 32'h100 : 32'h0)));
    endtask

    task automatic do_mzero();
        send_cmd(build_mxu_cmd(MXU_MZERO, 1'b1, 5'd0, 5'd0, 5'd0, 32'd0));
    endtask

    // MLOAD_W: 逐拍发送, 每拍一条指令
    // vs2_base: VRF 中权重起始寄存器号
    // num_regs: 需要加载的寄存器数 (= Tk, 因为每个寄存器存一行 k 的 16B)
    task automatic do_mload_w_all(input int vs2_base, input int num_regs);
        for (int k = 0; k < num_regs; k++)
            send_cmd(build_mxu_cmd(MXU_MLOAD_W,
                                   (k == num_regs - 1) ? 1'b0 : 1'b1,  // vm=0 表示 last
                                   (vs2_base + k),
                                   5'd0, 5'd0, 32'd0));
    endtask

    task automatic do_mload_a_per_row(input int vs2_base, input int Tk);
        int num_chunks;
        int total_beats;
        num_chunks = Tk / 16;
        if (num_chunks < 1) num_chunks = 1;
        total_beats = 16 * num_chunks;
        for (int beat = 0; beat < total_beats; beat++)
            send_cmd(build_mxu_cmd(MXU_MLOAD_A,
                                   (beat == total_beats - 1) ? 1'b0 : 1'b1,  // vm=0 last
                                   (vs2_base + beat),
                                   5'd0, 5'd0, 32'd0));
    endtask

    task automatic do_mma();
        send_cmd(build_mxu_cmd(MXU_MMA, 1'b1, 5'd0, 5'd0, 5'd0, 32'd0));
    endtask

    task automatic do_mfence();
        send_cmd(build_mxu_cmd(MXU_MFENCE, 1'b1, 5'd0, 5'd0, 5'd0, 32'd0));
    endtask

    task automatic do_mstore_one(input int vd_reg = 16);
        send_cmd(build_mstore_cmd(vd_reg[4:0]));
    endtask

    task automatic do_mstore_all();
        for (int i = 0; i < 64; i++)
            do_mstore_one(16);
    endtask

    // =========================================================
    // 软件参考模型: int8 矩阵乘 C[16][16] = A[16][Tk] * W[Tk][16]
    // =========================================================
    function automatic void ref_matmul(
        input  int Tk,
        input  byte signed weight[0:127][0:15],
        input  byte signed act[0:15][0:127],
        output int          result[0:15][0:15]
    );
        for (int m = 0; m < 16; m++)
            for (int n = 0; n < 16; n++) begin
                result[m][n] = 0;
                for (int k = 0; k < Tk; k++)
                    result[m][n] += int'(act[m][k]) * int'(weight[k][n]);
            end
    endfunction

    // =========================================================
    // 从捕获的 MSTORE 数据中提取 int32 结果
    // =========================================================
    function automatic void extract_results(
        input  int base_idx,
        output int result[0:15][0:15]
    );
        for (int row = 0; row < 16; row++)
            for (int col_chunk = 0; col_chunk < 4; col_chunk++) begin
                int idx = base_idx + row * 4 + col_chunk;
                logic [127:0] d = captured_data[idx];
                for (int j = 0; j < 4; j++)
                    result[row][col_chunk*4 + j] = $signed(d[j*32 +: 32]);
            end
    endfunction

    // =========================================================
    // 数据验证
    // =========================================================
    task automatic verify_results(
        input string test_name,
        input int    base_capture_idx,
        input int    expected[0:15][0:15],
        input int    valid_rows,
        input int    valid_cols
    );
        int actual[0:15][0:15];
        int mismatch_count;
        extract_results(base_capture_idx, actual);
        mismatch_count = 0;
        for (int m = 0; m < valid_rows; m++)
            for (int n = 0; n < valid_cols; n++) begin
                if (actual[m][n] !== expected[m][n]) begin
                    if (mismatch_count < 10)
                        $display("  MISMATCH [%0d][%0d]: expected=%0d actual=%0d",
                                 m, n, expected[m][n], actual[m][n]);
                    mismatch_count++;
                end
            end
        if (mismatch_count == 0)
            check({test_name, " data"}, 1'b1);
        else begin
            $display("  Total mismatches: %0d / %0d", mismatch_count, valid_rows*valid_cols);
            check({test_name, " data"}, 1'b0);
        end
    endtask

    // =========================================================
    // VRF 数据准备辅助 task
    //
    // 权重排布: v[vs2_base + k] 存 wbuf[k][0..15]
    //   每个 VRF 寄存器 128 位 = 16 字节, byte[n] = weight[k][n]
    //
    // 激活排布 (Tk=16, num_chunks=1):
    //   v[vs2_base + m] 存 abuf[m][0..15]
    //   byte[k] = act[m][k]
    //
    // 激活排布 (Tk=32, num_chunks=2):
    //   v[vs2_base + m*2 + chunk] 存 abuf[m][chunk*16 .. chunk*16+15]
    //
    // 通用: v[vs2_base + m * num_chunks + chunk]
    // =========================================================
    task automatic load_weight_to_vrf(
        input int vs2_base,
        input int Tk,
        input byte signed w_data[0:127][0:15]
    );
        byte unsigned vrf_bytes[16];
        for (int k = 0; k < Tk; k++) begin
            for (int n = 0; n < 16; n++)
                vrf_bytes[n] = w_data[k][n];
            vrf_write_bytes(vs2_base + k, vrf_bytes);
        end
    endtask

    task automatic load_act_to_vrf(
        input int vs2_base,
        input int Tk,
        input byte signed a_data[0:15][0:127]
    );
        byte unsigned vrf_bytes[16];
        int num_chunks;
        num_chunks = Tk / 16;
        if (num_chunks < 1) num_chunks = 1;
        for (int m = 0; m < 16; m++) begin
            for (int chunk = 0; chunk < num_chunks; chunk++) begin
                for (int j = 0; j < 16; j++)
                    vrf_bytes[j] = a_data[m][chunk * 16 + j];
                vrf_write_bytes(vs2_base + m * num_chunks + chunk, vrf_bytes);
            end
        end
    endtask

    // =========================================================
    // 完整的 MXU 计算流程封装
    // =========================================================
    task automatic run_mxu_pipeline(
        input int Tk,
        input int w_vrf_base,
        input int a_vrf_base,
        input bit do_zero
    );
        do_mload_w_all(w_vrf_base, Tk);
        do_mfence();
        if (do_zero)
            do_mzero();
        do_mload_a_per_row(a_vrf_base, Tk);  // 改这里
        do_mma();
        do_mfence();
    endtask

    // =========================================================
    // 测试主体
    // =========================================================
    initial begin
        $dumpfile("mxu_top_tb.vcd");
        $dumpvars(0, mxu_top_tb);

        rst_n              = 1'b0;
        insts_valid_rvs2cq = '0;
        insts_rvs2cq       = '0;
        retire_cnt = 0; mstore_cnt = 0; capture_idx = 0;
        total_pass = 0; total_fail = 0;
        tb_vrf_wr_en = 1'b0;

        repeat(10) @(posedge clk);
        rst_n = 1'b1;
        repeat(5) @(posedge clk);

        // =====================================================
        // DV1: MZERO 验证 — 累加器清零后 MSTORE 应全为 0
        // =====================================================
        begin
            int expected[0:15][0:15];
            $display("\n===== DV1: MZERO -> MSTORE all zeros =====");
            reset_cnt();

            do_mcfg(16);
            do_mzero();
            do_mstore_all();
            wait_idle();

            for (int m = 0; m < 16; m++)
                for (int n = 0; n < 16; n++)
                    expected[m][n] = 0;

            check("DV1: 64 mstore", mstore_cnt == 64);
            verify_results("DV1", 0, expected, 16, 16);
        end

        // =====================================================
        // DV2: 单位矩阵权重, 单位激活 — 验证 identity
        //
        // W = I (16x16), A = I (16x16), Tk=16
        // 期望 C = I
        // =====================================================
        begin
            byte signed w_data[0:127][0:15];
            byte signed a_data[0:15][0:127];
            int expected[0:15][0:15];

            $display("\n===== DV2: Identity W * Identity A (Tk=16) =====");
            reset_cnt();

            for (int k = 0; k < 128; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = 0;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 128; k++)
                    a_data[m][k] = 0;

            for (int i = 0; i < 16; i++) begin
                w_data[i][i] = 1;
                a_data[i][i] = 1;
            end

            ref_matmul(16, w_data, a_data, expected);

            // 权重写入 VRF v0..v15
            load_weight_to_vrf(0, 16, w_data);
            // 激活写入 VRF v16..v31 (Tk=16, num_chunks=1, 共 16 个寄存器)
            load_act_to_vrf(16, 16, a_data);

            do_mcfg(16);
            run_mxu_pipeline(16, 0, 16, 1'b1);
            do_mstore_all();
            wait_idle();

            check("DV2: 64 mstore", mstore_cnt == 64);
            verify_results("DV2", 0, expected, 16, 16);
        end

        // =====================================================
        // DV3: 全 1 权重, 递增激活
        //
        // W[k][n] = 1, A[m][k] = m+1, Tk=16
        // 期望 C[m][n] = (m+1) * 16
        // =====================================================
        begin
            byte signed w_data[0:127][0:15];
            byte signed a_data[0:15][0:127];
            int expected[0:15][0:15];

            $display("\n===== DV3: All-1 W * row-constant A (Tk=16) =====");
            reset_cnt();

            for (int k = 0; k < 128; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = 0;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 128; k++)
                    a_data[m][k] = 0;

            for (int k = 0; k < 16; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = 1;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 16; k++)
                    a_data[m][k] = byte'(m + 1);

            ref_matmul(16, w_data, a_data, expected);
            load_weight_to_vrf(0, 16, w_data);
            load_act_to_vrf(16, 16, a_data);

            do_mcfg(16);
            run_mxu_pipeline(16, 0, 16, 1'b1);
            do_mstore_all();
            wait_idle();

            check("DV3: 64 mstore", mstore_cnt == 64);
            verify_results("DV3", 0, expected, 16, 16);
        end

        // =====================================================
        // DV4: 负数权重测试
        //
        // W[k][n] = -1, A[m][k] = 2, Tk=16
        // 期望 C[m][n] = -32
        // =====================================================
        begin
            byte signed w_data[0:127][0:15];
            byte signed a_data[0:15][0:127];
            int expected[0:15][0:15];

            $display("\n===== DV4: Negative weights (Tk=16) =====");
            reset_cnt();

            for (int k = 0; k < 128; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = 0;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 128; k++)
                    a_data[m][k] = 0;

            for (int k = 0; k < 16; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = -1;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 16; k++)
                    a_data[m][k] = 2;

            ref_matmul(16, w_data, a_data, expected);
            load_weight_to_vrf(0, 16, w_data);
            load_act_to_vrf(16, 16, a_data);

            do_mcfg(16);
            run_mxu_pipeline(16, 0, 16, 1'b1);
            do_mstore_all();
            wait_idle();

            check("DV4: 64 mstore", mstore_cnt == 64);
            verify_results("DV4", 0, expected, 16, 16);
        end

        // =====================================================
        // DV5: 随机数据 Tk=16
        // =====================================================
        begin
            byte signed w_data[0:127][0:15];
            byte signed a_data[0:15][0:127];
            int expected[0:15][0:15];
            int seed;

            $display("\n===== DV5: Random data Tk=16 =====");
            reset_cnt();

            for (int k = 0; k < 128; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = 0;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 128; k++)
                    a_data[m][k] = 0;

            seed = 42;
            for (int k = 0; k < 16; k++)
                for (int n = 0; n < 16; n++) begin
                    seed = seed * 1103515245 + 12345;
                    w_data[k][n] = byte'((seed >> 16) & 8'hFF);
                end
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 16; k++) begin
                    seed = seed * 1103515245 + 12345;
                    a_data[m][k] = byte'((seed >> 16) & 8'hFF);
                end

            ref_matmul(16, w_data, a_data, expected);
            load_weight_to_vrf(0, 16, w_data);
            load_act_to_vrf(16, 16, a_data);

            do_mcfg(16);
            run_mxu_pipeline(16, 0, 16, 1'b1);
            do_mstore_all();
            wait_idle();

            check("DV5: 64 mstore", mstore_cnt == 64);
            verify_results("DV5", 0, expected, 16, 16);
        end

        // =====================================================
        // DV6: Tk=32 — 多 chunk 激活加载
        //
        // W[k][n] = 1, A[m][k] = 1, Tk=32
        // 期望 C[m][n] = 32
        //
        // 激活 VRF 排布: 16 行 * 2 chunks = 32 个寄存器
        //   v[base + m*2 + chunk]
        // =====================================================
        begin
            byte signed w_data[0:127][0:15];
            byte signed a_data[0:15][0:127];
            int expected[0:15][0:15];

            $display("\n===== DV6: Tk=32 multi-chunk activation =====");
            reset_cnt();

            for (int k = 0; k < 128; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = 0;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 128; k++)
                    a_data[m][k] = 0;

            for (int k = 0; k < 32; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = 1;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 32; k++)
                    a_data[m][k] = 1;

            ref_matmul(32, w_data, a_data, expected);

            // 权重: v0..v31 (32 个寄存器)
            load_weight_to_vrf(0, 32, w_data);
            // 激活: v32..v63 (16 行 * 2 chunks = 32 个寄存器)
            load_act_to_vrf(32, 32, a_data);

            do_mcfg(32);
            run_mxu_pipeline(32, 0, 32, 1'b1);
            do_mstore_all();
            wait_idle();

            check("DV6: 64 mstore", mstore_cnt == 64);
            verify_results("DV6", 0, expected, 16, 16);
        end

        // =====================================================
        // DV7: 累加验证 — 两次 MMA 不清零
        //
        // W=1, A[m][k]=m+1, Tk=16
        // 两次 MMA 不 MZERO => 结果 = 2 * 单次
        // =====================================================
        begin
            byte signed w_data[0:127][0:15];
            byte signed a_data[0:15][0:127];
            int single_expected[0:15][0:15];
            int double_expected[0:15][0:15];

            $display("\n===== DV7: Double MMA accumulation =====");
            reset_cnt();

            for (int k = 0; k < 128; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = 0;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 128; k++)
                    a_data[m][k] = 0;

            for (int k = 0; k < 16; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = 1;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 16; k++)
                    a_data[m][k] = byte'(m + 1);

            ref_matmul(16, w_data, a_data, single_expected);
            for (int m = 0; m < 16; m++)
                for (int n = 0; n < 16; n++)
                    double_expected[m][n] = single_expected[m][n] * 2;

            load_weight_to_vrf(0, 16, w_data);
            load_act_to_vrf(16, 16, a_data);

            do_mcfg(16);

            // 加载权重
            do_mload_w_all(0, 16);
            do_mfence();

            // 第一次 MMA
            do_mzero();
            do_mload_a_per_row(16, 16);
            do_mma();
            do_mfence();

            // 第二次 MMA — 不清零, 累加
            do_mload_a_per_row(16, 16);
            do_mma();
            do_mfence();

            do_mstore_all();
            wait_idle();

            check("DV7: 64 mstore", mstore_cnt == 64);
            verify_results("DV7", 0, double_expected, 16, 16);
        end

        // =====================================================
        // DV8: MZERO 隔离验证
        //
        // 第一次 MMA + MSTORE, 然后 MZERO + 第二次 MMA + MSTORE
        // 第二次结果应等于单次结果
        // =====================================================
        begin
            byte signed w_data[0:127][0:15];
            byte signed a_data[0:15][0:127];
            int expected[0:15][0:15];

            $display("\n===== DV8: MZERO isolation =====");
            reset_cnt();

            for (int k = 0; k < 128; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = 0;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 128; k++)
                    a_data[m][k] = 0;

            for (int k = 0; k < 16; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = 3;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 16; k++)
                    a_data[m][k] = 2;

            ref_matmul(16, w_data, a_data, expected);
            load_weight_to_vrf(0, 16, w_data);
            load_act_to_vrf(16, 16, a_data);

            do_mcfg(16);
            do_mload_w_all(0, 16);
            do_mfence();

            // 第一次: 计算并 store (结果不关心)
            do_mzero();
            do_mload_a_per_row(16, 16);
            do_mma();
            do_mfence();
            do_mstore_all();

            // 第二次: MZERO 清零后重新计算
            do_mzero();
            do_mload_a_per_row(16, 16);
            do_mma();
            do_mfence();
            do_mstore_all();
            wait_idle();

            check("DV8: 128 mstore", mstore_cnt == 128);
            verify_results("DV8", 64, expected, 16, 16);
        end

        // =====================================================
        // DV9: 边界值 — 最大正数 × 最大正数
        //
        // W[k][n] = 127, A[m][k] = 127, Tk=16
        // 期望 C[m][n] = 127 * 127 * 16 = 258064
        // =====================================================
        begin
            byte signed w_data[0:127][0:15];
            byte signed a_data[0:15][0:127];
            int expected[0:15][0:15];

            $display("\n===== DV9: Max positive (Tk=16) =====");
            reset_cnt();

            for (int k = 0; k < 128; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = 0;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 128; k++)
                    a_data[m][k] = 0;

            for (int k = 0; k < 16; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = 127;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 16; k++)
                    a_data[m][k] = 127;

            ref_matmul(16, w_data, a_data, expected);
            load_weight_to_vrf(0, 16, w_data);
            load_act_to_vrf(16, 16, a_data);

            do_mcfg(16);
            run_mxu_pipeline(16, 0, 16, 1'b1);
            do_mstore_all();
            wait_idle();

            check("DV9: 64 mstore", mstore_cnt == 64);
            verify_results("DV9", 0, expected, 16, 16);
        end

        // =====================================================
        // DV10: 边界值 — 最小负数 × 最大正数
        //
        // W[k][n] = -128, A[m][k] = 127, Tk=16
        // 期望 C[m][n] = -128 * 127 * 16 = -260096
        // =====================================================
        begin
            byte signed w_data[0:127][0:15];
            byte signed a_data[0:15][0:127];
            int expected[0:15][0:15];

            $display("\n===== DV10: Min negative * max positive (Tk=16) =====");
            reset_cnt();

            for (int k = 0; k < 128; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = 0;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 128; k++)
                    a_data[m][k] = 0;

            for (int k = 0; k < 16; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = -128;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 16; k++)
                    a_data[m][k] = 127;

            ref_matmul(16, w_data, a_data, expected);
            load_weight_to_vrf(0, 16, w_data);
            load_act_to_vrf(16, 16, a_data);

            do_mcfg(16);
            run_mxu_pipeline(16, 0, 16, 1'b1);
            do_mstore_all();
            wait_idle();

            check("DV10: 64 mstore", mstore_cnt == 64);
            verify_results("DV10", 0, expected, 16, 16);
        end

        // =====================================================
        // DV11: 对角线权重 + 递增激活 — 验证列路由
        //
        // W = diag(1,2,...,16), A[m][k] = k+1, Tk=16
        // 期望 C[m][n] = (n+1)*(n+1)
        // =====================================================
        begin
            byte signed w_data[0:127][0:15];
            byte signed a_data[0:15][0:127];
            int expected[0:15][0:15];

            $display("\n===== DV11: Diagonal W * column-varying A (Tk=16) =====");
            reset_cnt();

            for (int k = 0; k < 128; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = 0;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 128; k++)
                    a_data[m][k] = 0;

            for (int i = 0; i < 16; i++)
                w_data[i][i] = byte'(i + 1);
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 16; k++)
                    a_data[m][k] = byte'(k + 1);

            ref_matmul(16, w_data, a_data, expected);
            load_weight_to_vrf(0, 16, w_data);
            load_act_to_vrf(16, 16, a_data);

            do_mcfg(16);
            run_mxu_pipeline(16, 0, 16, 1'b1);
            do_mstore_all();
            wait_idle();

            check("DV11: 64 mstore", mstore_cnt == 64);
            verify_results("DV11", 0, expected, 16, 16);
        end

        // =====================================================
        // DV12: MSTORE 数据时序验证
        //
        // W[k][n] = n, A[m][k] = 1, Tk=16
        // 期望 C[m][n] = n * 16 (所有行相同)
        // =====================================================
        begin
            byte signed w_data[0:127][0:15];
            byte signed a_data[0:15][0:127];
            int expected[0:15][0:15];

            $display("\n===== DV12: MSTORE timing (wrapper fix, Tk=16) =====");
            reset_cnt();

            for (int k = 0; k < 128; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = 0;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 128; k++)
                    a_data[m][k] = 0;

            for (int k = 0; k < 16; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = byte'(n);
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 16; k++)
                    a_data[m][k] = 1;

            ref_matmul(16, w_data, a_data, expected);
            load_weight_to_vrf(0, 16, w_data);
            load_act_to_vrf(16, 16, a_data);

            do_mcfg(16);
            run_mxu_pipeline(16, 0, 16, 1'b1);
            do_mstore_all();
            wait_idle();

            check("DV12: 64 mstore", mstore_cnt == 64);
            verify_results("DV12", 0, expected, 16, 16);

            // 额外检查: 第一条 MSTORE 不应为全 0
            begin
                logic [127:0] first_store = captured_data[0];
                bit first_nonzero = (first_store != 128'd0);
                check("DV12: first MSTORE not zero (timing fix)", first_nonzero);
                if (!first_nonzero)
                    $display("  first MSTORE data = 0x%032h (BUG: stale data)", first_store);
            end
        end

        // =====================================================
        // DV13: 连续两轮不同数据 — 验证状态不残留
        //
        // 第一轮: W=1, A=1 => C[m][n]=16
        // 第二轮: W=2, A=3 => C[m][n]=96
        // =====================================================
        begin
            byte signed w_data[0:127][0:15];
            byte signed a_data[0:15][0:127];
            int expected1[0:15][0:15];
            int expected2[0:15][0:15];

            $display("\n===== DV13: Two rounds different data =====");
            reset_cnt();

            // --- 第一轮 ---
            for (int k = 0; k < 128; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = 0;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 128; k++)
                    a_data[m][k] = 0;

            for (int k = 0; k < 16; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = 1;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 16; k++)
                    a_data[m][k] = 1;

            ref_matmul(16, w_data, a_data, expected1);
            load_weight_to_vrf(0, 16, w_data);
            load_act_to_vrf(16, 16, a_data);

            do_mcfg(16);
            run_mxu_pipeline(16, 0, 16, 1'b1);
            do_mstore_all();
            wait_idle();

            check("DV13 round1: 64 mstore", mstore_cnt == 64);
            verify_results("DV13 round1", 0, expected1, 16, 16);

            // --- 第二轮: 新数据 ---
            begin
                int saved_capture = capture_idx;

                for (int k = 0; k < 16; k++)
                    for (int n = 0; n < 16; n++)
                        w_data[k][n] = 2;
                for (int m = 0; m < 16; m++)
                    for (int k = 0; k < 16; k++)
                        a_data[m][k] = 3;

                ref_matmul(16, w_data, a_data, expected2);
                load_weight_to_vrf(0, 16, w_data);
                load_act_to_vrf(16, 16, a_data);

                retire_cnt = 0; mstore_cnt = 0;

                do_mload_w_all(0, 16);
                do_mfence();
                do_mzero();
                do_mload_a_per_row(16, 16);
                do_mma();
                do_mfence();
                do_mstore_all();
                wait_idle();

                check("DV13 round2: 64 mstore", mstore_cnt == 64);
                verify_results("DV13 round2", saved_capture, expected2, 16, 16);
            end
        end

        // =====================================================
        // DV14: Tk=64 — 大深度随机测试
        //
        // 权重 v0..v63, 激活 v64..v127 (16行 * 4chunks)
        // =====================================================
        begin
            byte signed w_data[0:127][0:15];
            byte signed a_data[0:15][0:127];
            int expected[0:15][0:15];
            int seed;

            $display("\n===== DV14: Random data Tk=64 =====");
            reset_cnt();

            for (int k = 0; k < 128; k++)
                for (int n = 0; n < 16; n++)
                    w_data[k][n] = 0;
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 128; k++)
                    a_data[m][k] = 0;

            seed = 12345;
            for (int k = 0; k < 64; k++)
                for (int n = 0; n < 16; n++) begin
                    seed = seed * 1103515245 + 12345;
                    w_data[k][n] = byte'((seed >> 16) & 8'hFF);
                end
            for (int m = 0; m < 16; m++)
                for (int k = 0; k < 64; k++) begin
                    seed = seed * 1103515245 + 12345;
                    a_data[m][k] = byte'((seed >> 16) & 8'hFF);
                end

            ref_matmul(64, w_data, a_data, expected);

            // 权重: v0..v63
            load_weight_to_vrf(0, 64, w_data);
            // 激活: v64..v127 (16 行 * 4 chunks = 64 个寄存器)
            load_act_to_vrf(64, 64, a_data);

            do_mcfg(64);
            run_mxu_pipeline(64, 0, 64, 1'b1);
            do_mstore_all();
            wait_idle();

            check("DV14: 64 mstore", mstore_cnt == 64);
            verify_results("DV14", 0, expected, 16, 16);
        end

        // =====================================================
        // 保留原有功能测试 (不验证数据)
        // =====================================================

        // TC-F1: 基础 MCFG + MZERO
        begin
            $display("\n===== TC-F1: MCFG + MZERO =====");
            reset_cnt();
            do_mcfg(16);
            do_mzero();
            wait_idle();
            check("TC-F1: retire>=2", retire_cnt >= 2);
        end

        // TC-F2: back-to-back MFENCE
        begin
            $display("\n===== TC-F2: back-to-back MFENCE =====");
            reset_cnt();
            do_mfence();
            do_mfence();
            do_mfence();
            wait_idle();
            check("TC-F2: retire>=3", retire_cnt >= 3);
        end

        // TC-F3: back-to-back MZERO
        begin
            $display("\n===== TC-F3: back-to-back MZERO =====");
            reset_cnt();
            do_mzero();
            do_mzero();
            do_mzero();
            wait_idle();
            check("TC-F3: retire>=3", retire_cnt >= 3);
        end

        // TC-F4: 128 consecutive MSTORE
        begin
            $display("\n===== TC-F4: 128 consecutive MSTORE =====");
            reset_cnt();
            for (int i = 0; i < 128; i++)
                do_mstore_one(16);
            wait_idle();
            check("TC-F4: 128 mstore", mstore_cnt == 128);
        end

        // =====================================================
        // 汇总
        // =====================================================
        #200;
        $display("\n========================================");
        $display("  PASS: %0d  FAIL: %0d", total_pass, total_fail);
        $display("========================================");
        if (total_fail > 0)
            $display("[RESULT] SOME TESTS FAILED");
        else
            $display("[RESULT] ALL TESTS PASSED");
        $finish;
    end

endmodule