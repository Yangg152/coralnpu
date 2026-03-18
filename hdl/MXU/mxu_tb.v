`timescale 1ns / 1ps

module mxu_tb;

localparam CLK_HALF = 5;

reg clk = 0;
always #CLK_HALF clk = ~clk;

// 信号声明
reg          rst_n;
reg  [2:0]   op_type;
reg          op_valid;
wire         op_ready;
wire         op_done;

reg  [7:0]   cfg_Tk;
reg          cfg_signed; 

reg  [127:0] weight_vec;
reg          weight_valid;
wire         weight_ready;

reg  [127:0] act_row [0:15];
reg          act_valid;
wire         act_ready;

wire [127:0] result_bus [0:15];
wire         result_valid;
wire         mfence_done;

// 显式连线实例化
mxu u_mxu (
    .clk          (clk),
    .rst_n        (rst_n),
    .op_type      (op_type),
    .op_valid     (op_valid),
    .op_ready     (op_ready),
    .op_done      (op_done),
    .cfg_Tk       (cfg_Tk),
    .cfg_signed   (cfg_signed),
    .weight_vec   (weight_vec),
    .weight_valid (weight_valid),
    .weight_ready (weight_ready),
    .act_row      (act_row),
    .act_valid    (act_valid),
    .act_ready    (act_ready),
    .result_bus   (result_bus),
    .result_valid (result_valid),
    .mfence_done  (mfence_done)
);

// 存储与统计
reg [127:0] weight_mem [0:127];
reg [127:0] act_mem    [0:127];
reg [127:0] golden_mem [0:255]; 

integer pass_cnt=0, fail_cnt=0;

// 操作码枚举
localparam [2:0]
    OP_MCFG    = 3'd0,  
    OP_MLOAD_W = 3'd1, 
    OP_MZERO   = 3'd2,
    OP_MMA     = 3'd3,  
    OP_MSTORE  = 3'd4, 
    OP_MFENCE  = 3'd5,
    OP_MLOAD_A = 3'd6;

// 测试Task
task send_op_task(input [2:0] op);
    begin
        @(posedge clk);
        while (!op_ready) @(posedge clk);
        @(negedge clk);
        op_type  = op;
        op_valid = 1'b1;
        @(posedge clk); #1;
        op_valid = 1'b0;
    end
endtask

task do_mload_w(input integer num_rows, input integer mem_off);
    integer k;
    begin
        send_op_task(OP_MLOAD_W);
        for (k = 0; k < num_rows; k = k + 1) begin
            @(negedge clk);
            while (!weight_ready) @(negedge clk);
            weight_vec   = weight_mem[mem_off + k];
            weight_valid = 1'b1;
            @(posedge clk); #1;
            weight_valid = 1'b0;
        end
        while (!op_done) @(posedge clk);
    end
endtask

task do_mload_a(input integer tk, input integer mem_off);
    integer c, m;
    begin
        send_op_task(OP_MLOAD_A);
        for (c = 0; c < (tk/16); c = c + 1) begin
            @(negedge clk);
            while (!act_ready) @(negedge clk);
            for (m = 0; m < 16; m = m + 1)
                act_row[m] = act_mem[mem_off + c*16 + m];
            act_valid = 1'b1;
            @(posedge clk); #1;
            act_valid = 1'b0;
        end
        while (!op_done) @(posedge clk);
    end
endtask

reg [127:0] captured [0:63];
task do_mstore_and_check(input [255:0] tc_label);
    integer cyc, b, mis;
    begin
        mis = 0;
        send_op_task(OP_MSTORE);
        cyc = 0;
        @(posedge clk);
        while (cyc < 4) begin
            if (result_valid) begin
                for (b = 0; b < 16; b = b + 1)
                    captured[cyc*16 + b] = result_bus[b];
                cyc = cyc + 1;
            end
            if (cyc < 4) @(posedge clk);
        end
        while (!op_done) @(posedge clk);

        for (b = 0; b < 64; b = b + 1) begin
            if (captured[b] !== golden_mem[b]) begin
                $display("[FAIL] TC%0s bus[%0d]: got=%032x exp=%032x", tc_label, b, captured[b], golden_mem[b]);
                mis = mis + 1;
            end
        end
        if (mis == 0) begin $display("[PASS] TC%0s 匹配!", tc_label); pass_cnt++; end
        else begin fail_cnt++; end
    end
endtask

integer init_i;

initial begin
    // 强制初始化所有驱动信号
    rst_n        = 0; 
    op_type      = 0;
    op_valid     = 0; 
    cfg_Tk       = 8'd64;
    cfg_signed   = 1'b1;
    weight_vec   = 128'd0;
    weight_valid = 0; 
    act_valid    = 0;
    for (init_i=0; init_i<16; init_i=init_i+1) act_row[init_i] = 128'd0;

    repeat(5) @(posedge clk);
    rst_n = 1;
    repeat(3) @(posedge clk);

    $display("========================================");
    $display("  MMU 阵列测试平台启动");
    $display("========================================");

    // ---------------------------------------------------------
    // TC1: 基础随机用例 (Tk=16)
    // ---------------------------------------------------------
    $display("\n--- TC1: Tk=16, 基础随机测试 ---");
    $readmemh("cases/tc1_weight.hex", weight_mem);
    $readmemh("cases/tc1_act.hex",    act_mem);
    $readmemh("cases/tc1_golden.hex", golden_mem);
    cfg_Tk = 8'd16; send_op_task(OP_MCFG);
    send_op_task(OP_MZERO);
    do_mload_w(16, 0);
    do_mload_a(16, 0);
    send_op_task(OP_MMA); while(!op_done) @(posedge clk);
    do_mstore_and_check("1");

    // ---------------------------------------------------------
    // TC2: 深度随机用例 (Tk=64)
    // ---------------------------------------------------------
    $display("\n--- TC2: Tk=64, 深度随机测试 ---");
    $readmemh("cases/tc2_weight.hex", weight_mem);
    $readmemh("cases/tc2_act.hex",    act_mem);
    $readmemh("cases/tc2_golden.hex", golden_mem);
    cfg_Tk = 8'd64; send_op_task(OP_MCFG);
    send_op_task(OP_MZERO);
    do_mload_w(64, 0);
    do_mload_a(64, 0);
    send_op_task(OP_MMA); while(!op_done) @(posedge clk);
    do_mstore_and_check("2");

    // ---------------------------------------------------------
    // TC3: 边界值测试 (-128 * 127) 检查符号扩展
    // ---------------------------------------------------------
    $display("\n--- TC3: Tk=16, 极值符号扩展测试 ---");
    $readmemh("cases/tc3_weight.hex", weight_mem);
    $readmemh("cases/tc3_act.hex",    act_mem);
    $readmemh("cases/tc3_golden.hex", golden_mem);
    cfg_Tk = 8'd16; send_op_task(OP_MCFG);
    send_op_task(OP_MZERO);
    do_mload_w(16, 0);
    do_mload_a(16, 0);
    send_op_task(OP_MMA); while(!op_done) @(posedge clk);
    do_mstore_and_check("3");

    // ---------------------------------------------------------
    // TC4: 全零测试 (A矩阵为0)
    // ---------------------------------------------------------
    $display("\n--- TC4: Tk=16, 零矩阵乘法测试 ---");
    $readmemh("cases/tc4_weight.hex", weight_mem);
    $readmemh("cases/tc4_act.hex",    act_mem);
    $readmemh("cases/tc4_golden.hex", golden_mem);
    cfg_Tk = 8'd16; send_op_task(OP_MCFG);
    send_op_task(OP_MZERO);
    do_mload_w(16, 0);
    do_mload_a(16, 0);
    send_op_task(OP_MMA); while(!op_done) @(posedge clk);
    do_mstore_and_check("4");

    // ---------------------------------------------------------
    // TC5: 多次连续累加测试 (无MZERO) - 模拟分块矩阵乘
    // ---------------------------------------------------------
    $display("\n--- TC5: Tk=64, 连续4次累加测试 (不清零) ---");
    cfg_Tk = 8'd64; send_op_task(OP_MCFG);
    send_op_task(OP_MZERO); // 仅在初始清零，后续直接累加

    // Tile 0
    $readmemh("cases/tc5_w0.hex", weight_mem); $readmemh("cases/tc5_a0.hex", act_mem);
    do_mload_w(64, 0); do_mload_a(64, 0); send_op_task(OP_MMA); while(!op_done) @(posedge clk);
    
    // Tile 1
    $readmemh("cases/tc5_w1.hex", weight_mem); $readmemh("cases/tc5_a1.hex", act_mem);
    do_mload_w(64, 0); do_mload_a(64, 0); send_op_task(OP_MMA); while(!op_done) @(posedge clk);
    
    // Tile 2
    $readmemh("cases/tc5_w2.hex", weight_mem); $readmemh("cases/tc5_a2.hex", act_mem);
    do_mload_w(64, 0); do_mload_a(64, 0); send_op_task(OP_MMA); while(!op_done) @(posedge clk);
    
    // Tile 3
    $readmemh("cases/tc5_w3.hex", weight_mem); $readmemh("cases/tc5_a3.hex", act_mem);
    do_mload_w(64, 0); do_mload_a(64, 0); send_op_task(OP_MMA); while(!op_done) @(posedge clk);

    $readmemh("cases/tc5_golden.hex", golden_mem);
    do_mstore_and_check("5");


    repeat(5) @(posedge clk);
    $display("\n========================================");
    if (fail_cnt == 0) begin
        $display("  所有用例均通过！(总计 %0d 组)", pass_cnt);
    end else begin
        $display("  通过: %0d  失败: %0d", pass_cnt, fail_cnt);
    end
    $display("========================================");
    $finish;
end

initial begin
    #5_000_000;  // 增加超时时间以容纳更长的测试
    $display("[ERROR] 仿真超时！");
    $finish;
end

// initial begin
//     $dumpfile("mxu_wave.vcd");
//     $dumpvars(0, mxu_tb);
// end

initial begin
    $fsdbDumpfile("FSDB/mxu_wave.fsdb");
    $fsdbDumpvars(0, mxu_tb);
    $fsdbDumpMDA(); // 这一句非常重要，用于导出多维数组(Memory/Array)的波形
end

endmodule
