module Sram_256x128(
  input          clock,
  input          enable,
  input          write,
  input  [7:0]   addr,       // 256 深度 → 8-bit 地址
  input  [127:0] wdata,
  input  [15:0]  wmask,      // 16 个 byte 的写掩码
  output [127:0] rdata
);

`ifdef USE_TSMC12FFC
  // TODO: 替换为对应的 TSMC12FFC 256x128 macro
`elsif USE_GF22
  // TODO: 替换为对应的 GF22 256x128 macro
`else
///////////////////////////
////// Generic SRAM ///////
///////////////////////////
  reg [127:0] mem [0:255];
  reg [7:0]   raddr;

  assign rdata = mem[raddr];

`ifndef SYNTHESIS
  task randomMemoryAll;
    for (int i = 0; i < 256; i++) begin
      mem[i] = { $random, $random, $random, $random };
    end
  endtask

  initial begin
    randomMemoryAll;
  end
`endif

  always @(posedge clock) begin
    for (int i = 0; i < 16; i++) begin
      if (enable & write & wmask[i]) begin
        mem[addr][i*8 +: 8] <= wdata[8*i +: 8];
      end
    end

    if (enable & ~write) begin
      raddr <= addr;
    end
  end
`endif

endmodule