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

module i2c_master_top
  import i2c_master_pkg::*;
  import tlul_pkg::*;
  import coralnpu_tlul_pkg_32::*;
#(
    parameter int FifoDepth = 16
) (
    input clk_i,
    input rst_ni,

    // TileLink-UL Interface
    input  coralnpu_tlul_pkg_32::tl_h2d_t tl_i,
    output coralnpu_tlul_pkg_32::tl_d2h_t tl_o,

    // I2C Interface
    input scl_i,
    output logic scl_o,
    output logic scl_en_o,
    input sda_i,
    output logic sda_o,
    output logic sda_en_o
);

  // --- Integrity Checking (A Channel) ---
  logic [6:0] calc_cmd_intg;
  logic [6:0] calc_data_intg;
  logic intg_error;

  // Cat(instr_type, address, opcode, mask)
  // 4 + 32 + 3 + 4 = 43 bits. Zero pad to 57 bits.
  logic [56:0] cmd_data;
  assign cmd_data = {14'h0, tl_i.a_user.instr_type, tl_i.a_address, tl_i.a_opcode, tl_i.a_mask};

  assign calc_cmd_intg = secded_inv_64_57_enc(cmd_data);
  assign calc_data_intg = secded_inv_39_32_enc(tl_i.a_data);

  assign intg_error = tl_i.a_valid & ((calc_cmd_intg != tl_i.a_user.cmd_intg) | 
                                      (calc_data_intg != tl_i.a_user.data_intg));

  always_ff @(posedge clk_i) begin
    if (tl_i.a_valid && tl_o_core.a_ready && intg_error) begin
      $display("[%0t] Integrity Error Detected!", $time);
      $display("  Addr: 0x%h, Opcode: %d, Mask: 0x%h, Data: 0x%h", tl_i.a_address, tl_i.a_opcode,
               tl_i.a_mask, tl_i.a_data);
      $display("  InstrType: 0x%h", tl_i.a_user.instr_type);
      $display("  CmdIntg: Got 0x%h, Expected 0x%h (cmd_data=0x%h)", tl_i.a_user.cmd_intg,
               calc_cmd_intg, cmd_data);
      $display("  DataIntg: Got 0x%h, Expected 0x%h", tl_i.a_user.data_intg, calc_data_intg);
    end
  end

  // --- Core Instance ---
  coralnpu_tlul_pkg_32::tl_d2h_t tl_o_core;
  i2c_master #(
      .FifoDepth(FifoDepth)
  ) i_core (
      .clk_i,
      .rst_ni,
      .tl_i,
      .tl_o(tl_o_core),
      .scl_i,
      .scl_o,
      .scl_en_o,
      .sda_i,
      .sda_o,
      .sda_en_o,
      .intr_o(),  // Not used at top level for now
      .intg_error_i(intg_error)
  );

  // --- Integrity Generation (D Channel) ---
  logic [ 6:0] gen_rsp_intg;
  logic [ 6:0] gen_data_intg;

  // Cat(opcode, size, error)
  // 3 + 2 + 1 = 6 bits. Zero pad to 57 bits.
  logic [56:0] rsp_data;
  assign rsp_data = {51'h0, tl_o_core.d_opcode, tl_o_core.d_size, tl_o_core.d_error};

  assign gen_rsp_intg = secded_inv_64_57_enc(rsp_data);
  assign gen_data_intg = secded_inv_39_32_enc(tl_o_core.d_data);

  always_comb begin
    tl_o = tl_o_core;
    tl_o.d_user.rsp_intg = gen_rsp_intg;
    tl_o.d_user.data_intg = gen_data_intg;
  end

endmodule
