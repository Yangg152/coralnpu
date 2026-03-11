// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package i2c_master_pkg;

  // Register Offsets
  localparam logic [11:0] INTR_STATE = 12'h000;
  localparam logic [11:0] INTR_ENABLE = 12'h004;
  localparam logic [11:0] CTRL = 12'h008;
  localparam logic [11:0] STATUS = 12'h00C;
  localparam logic [11:0] FDATA = 12'h010;
  localparam logic [11:0] FIFO_CTRL = 12'h014;
  localparam logic [11:0] CLK_DIV = 12'h018;

  // I2C Command Bits for FDATA Write
  localparam int FDATA_START = 8;
  localparam int FDATA_STOP = 9;
  localparam int FDATA_READ = 10;

  typedef enum logic [2:0] {
    Idle,
    Start,
    Addr,
    AckAddr,
    Write,
    Read,
    AckData,
    Stop
  } i2c_state_e;

  // SECDED Functions for Integrity
  function automatic logic [6:0] secded_inv_39_32_enc(logic [31:0] data);
    logic [6:0] ecc;
    ecc[0] = ^(data & 32'h002606BD);  // Note: Masks in python were > 32 bit, but data is 32.
                                      // Re-checking python: 0x002606BD25 is 40 bits? 
                                      // Actually it's 32-bit data. The masks are 32-bit.
    // Let's use the masks from TlulIntegrity.scala (implicitly) or secded_golden.py
    // Wait, secded_golden.py has: p0 = _parity(data_o & 0x002606BD25)
    // 0x002606BD25 is 37 bits? No, 10 hex digits = 40 bits.
    // If data is 32 bits, the top bits of mask are ignored.
    ecc[0] = ^(data & 32'h2606BD25);
    ecc[1] = ^(data & 32'hDEBA8050);
    ecc[2] = ^(data & 32'h413D89AA);
    ecc[3] = ^(data & 32'h31234ED1);
    ecc[4] = ^(data & 32'hC2C1323B);
    ecc[5] = ^(data & 32'h2DCC624C);
    ecc[6] = ^(data & 32'h98505586);
    return ecc ^ 7'h2A;  // Inversion constant for 39_32 is 0x2A shifted? 
                         // Chisel says: data_o.asUInt ^ "h2A00000000".U
                         // That means XOR the ECC bits with 0x2A.
  endfunction

  function automatic logic [6:0] secded_inv_64_57_enc(logic [56:0] data);
    logic [6:0] ecc;
    ecc[0] = ^(data & 57'h0103FFF800007FFF);
    ecc[1] = ^(data & 57'h017C1FF801FF801F);
    ecc[2] = ^(data & 57'h01BDE1F87E0781E1);
    ecc[3] = ^(data & 57'h01DEEE3B8E388E22);
    ecc[4] = ^(data & 57'h01EF76CDB2C93244);
    ecc[5] = ^(data & 57'h01F7BB56D5525488);
    ecc[6] = ^(data & 57'h01FBDDA769A46910);
    return ecc ^ 7'h2a;  // OpenTitan/Chisel use 0x2a for the 7-bit ECC field
  endfunction

endpackage
