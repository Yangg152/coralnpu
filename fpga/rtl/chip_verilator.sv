// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

module chip_verilator
    #(parameter MemInitFile = "",
      parameter int ClockFrequencyMhz = 80)
    (input clk_i,
     input rst_ni,
     input prim_mubi_pkg::mubi4_t scanmode_i,
     input top_pkg::uart_sideband_i_t[1 : 0] uart_sideband_i,
     output top_pkg::uart_sideband_o_t[1 : 0] uart_sideband_o);

  logic sck, csb, mosi, miso;

  spi_dpi_master i_spi_dpi_master (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .sck_o(sck),
    .csb_o(csb),
    .mosi_o(mosi),
    .miso_i(miso)
  );

  logic spim_sclk, spim_csb, spim_mosi, spim_miso;

  display_dpi i_display_dpi (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .sck_i(spim_sclk),
    .csb_i(spim_csb),
    .mosi_i(spim_mosi),
    .dc_i(gpio_o[0]),
    .rst_i(gpio_o[1]),
    .miso_o(spim_miso)
  );

  wire [7:0] gpio_o;
  wire [7:0] gpio_en_o;
  wire [7:0] gpio_i;

  gpio_dpi i_gpio_dpi (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .gpio_o(gpio_o),
    .gpio_en_o(gpio_en_o),
    .gpio_i(gpio_i)
  );

  logic uart0_rx;
  logic uart0_tx;

  uartdpi #(.BAUD(115200),
            .FREQ(ClockFrequencyMhz * 1_000_000),
            .NAME("uart0"),
            .EXIT_STRING("EXIT"))
      i_uartdpi0(.clk_i(clk_i),
                 .rst_ni(rst_ni),
                 .active(1'b1),
                 .tx_o(uart0_rx),
                 .rx_i(uart0_tx));

  logic uart1_rx;
  logic uart1_tx;

  uartdpi #(.BAUD(115200),
            .FREQ(ClockFrequencyMhz * 1_000_000),
            .NAME("uart1"),
            .EXIT_STRING("EXIT"))
      i_uartdpi1(.clk_i(clk_i),
                 .rst_ni(rst_ni),
                 .active(1'b1),
                 .tx_o(uart1_rx),
                 .rx_i(uart1_tx));

  assign uart0_tx = uart_sideband_o[0].cio_tx;
  assign uart1_tx = uart_sideband_o[1].cio_tx;

  logic tck_i, tms_i, trst_ni, td_i, td_o;
  jtagdpi i_jtagdpi (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .active(1'b1),
    .jtag_tck(tck_i),
    .jtag_tms(tms_i),
    .jtag_tdi(td_i),
    .jtag_tdo(td_o),
    .jtag_trst_n(trst_ni),
    .jtag_srst_n()
  );

  logic dm_req_valid, dm_req_ready;
  dm::dmi_req_t dm_req;
  logic dm_rsp_valid, dm_rsp_ready;
  dm::dmi_resp_t dm_rsp;
  logic dmi_rst_n;

  dmi_jtag #(.IdcodeValue(32'h04f5484d)) i_jtag (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .testmode_i(1'b0),
    .test_rst_ni(1'b1),
    .dmi_rst_no(dmi_rst_n),
    .dmi_req_o(dm_req),
    .dmi_req_valid_o(dm_req_valid),
    .dmi_req_ready_i(dm_req_ready),
    .dmi_resp_i(dm_rsp),
    .dmi_resp_ready_o(dm_rsp_ready),
    .dmi_resp_valid_i(dm_rsp_valid),
    .tck_i(tck_i),
    .tms_i(tms_i),
    .trst_ni(trst_ni),
    .td_i(td_i),
    .td_o(td_o),
    .tdo_oe_o(/*tdo_oe_o*/)
  );

  logic scl_i, scl_o, scl_en_o;
  logic sda_i, sda_o, sda_en_o;

  wire scl_bus, sda_bus;
  assign scl_bus = scl_en_o ? 1'b0 : 1'bz;
  assign (pull1, pull0) scl_bus = 1'b1;

  assign sda_bus = sda_en_o ? 1'b0 : 1'bz;
  assign sda_bus = hm_sda_en ? 1'b0 : 1'bz;
  assign (pull1, pull0) sda_bus = 1'b1;

  assign scl_i = scl_bus;
  assign sda_i = sda_bus;

  logic hm_sda_en;
  hm01b0_model i_hm01b0 (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .scl_i(scl_bus),
    .sda_i(sda_bus),
    .sda_en_o(hm_sda_en)
  );

  logic io_halted_o;
  logic io_fault_o;
  coralnpu_soc #(.MemInitFile(MemInitFile),
               .ClockFrequencyMhz(ClockFrequencyMhz))
    i_coralnpu_soc (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .spi_clk_i(sck),
      .spi_csb_i(csb),
      .spi_mosi_i(mosi),
      .spi_miso_o(miso),
      .spim_sclk_o(spim_sclk),
      .spim_csb_o(spim_csb),
      .spim_mosi_o(spim_mosi),
      .spim_miso_i(spim_miso),
      .spim_clk_i(clk_i),
      .gpio_o(gpio_o),
      .gpio_en_o(gpio_en_o),
      .gpio_i(gpio_i),
      .scanmode_i('0),
      .uart_sideband_i(uart_sideband_i),
      .uart_sideband_o(uart_sideband_o),
      .scl_i(scl_i),
      .scl_o(scl_o),
      .scl_en_o(scl_en_o),
      .sda_i(sda_i),
      .sda_o(sda_o),
      .sda_en_o(sda_en_o),
      .io_halted(io_halted_o),
      .io_fault(io_fault_o),
      .ddr_clk_i(1'b0),
      .ddr_rst(1'b0),
      .io_ddr_ctrl_axi_aw_valid(),
      .io_ddr_ctrl_axi_aw_ready(1'b0),
      .io_ddr_ctrl_axi_aw_bits_addr(),
      .io_ddr_ctrl_axi_aw_bits_prot(),
      .io_ddr_ctrl_axi_aw_bits_id(),
      .io_ddr_ctrl_axi_aw_bits_len(),
      .io_ddr_ctrl_axi_aw_bits_size(),
      .io_ddr_ctrl_axi_aw_bits_burst(),
      .io_ddr_ctrl_axi_aw_bits_lock(),
      .io_ddr_ctrl_axi_aw_bits_cache(),
      .io_ddr_ctrl_axi_aw_bits_qos(),
      .io_ddr_ctrl_axi_aw_bits_region(),
      .io_ddr_ctrl_axi_w_valid(),
      .io_ddr_ctrl_axi_w_ready(1'b0),
      .io_ddr_ctrl_axi_w_bits_data(),
      .io_ddr_ctrl_axi_w_bits_last(),
      .io_ddr_ctrl_axi_w_bits_strb(),
      .io_ddr_ctrl_axi_b_valid(1'b0),
      .io_ddr_ctrl_axi_b_ready(),
      .io_ddr_ctrl_axi_b_bits_id(6'b0),
      .io_ddr_ctrl_axi_b_bits_resp(2'b0),
      .io_ddr_ctrl_axi_ar_valid(),
      .io_ddr_ctrl_axi_ar_ready(1'b0),
      .io_ddr_ctrl_axi_ar_bits_addr(),
      .io_ddr_ctrl_axi_ar_bits_prot(),
      .io_ddr_ctrl_axi_ar_bits_id(),
      .io_ddr_ctrl_axi_ar_bits_len(),
      .io_ddr_ctrl_axi_ar_bits_size(),
      .io_ddr_ctrl_axi_ar_bits_burst(),
      .io_ddr_ctrl_axi_ar_bits_lock(),
      .io_ddr_ctrl_axi_ar_bits_cache(),
      .io_ddr_ctrl_axi_ar_bits_qos(),
      .io_ddr_ctrl_axi_ar_bits_region(),
      .io_ddr_ctrl_axi_r_valid(1'b0),
      .io_ddr_ctrl_axi_r_ready(),
      .io_ddr_ctrl_axi_r_bits_data(32'b0),
      .io_ddr_ctrl_axi_r_bits_id(6'b0),
      .io_ddr_ctrl_axi_r_bits_resp(2'b0),
      .io_ddr_ctrl_axi_r_bits_last(1'b0),
      .io_ddr_mem_axi_aw_valid(),
      .io_ddr_mem_axi_aw_ready(1'b0),
      .io_ddr_mem_axi_aw_bits_addr(),
      .io_ddr_mem_axi_aw_bits_prot(),
      .io_ddr_mem_axi_aw_bits_id(),
      .io_ddr_mem_axi_aw_bits_len(),
      .io_ddr_mem_axi_aw_bits_size(),
      .io_ddr_mem_axi_aw_bits_burst(),
      .io_ddr_mem_axi_aw_bits_lock(),
      .io_ddr_mem_axi_aw_bits_cache(),
      .io_ddr_mem_axi_aw_bits_qos(),
      .io_ddr_mem_axi_aw_bits_region(),
      .io_ddr_mem_axi_w_valid(),
      .io_ddr_mem_axi_w_ready(1'b0),
      .io_ddr_mem_axi_w_bits_data(),
      .io_ddr_mem_axi_w_bits_last(),
      .io_ddr_mem_axi_w_bits_strb(),
      .io_ddr_mem_axi_b_valid(1'b0),
      .io_ddr_mem_axi_b_ready(),
      .io_ddr_mem_axi_b_bits_id(6'b0),
      .io_ddr_mem_axi_b_bits_resp(2'b0),
      .io_ddr_mem_axi_ar_valid(),
      .io_ddr_mem_axi_ar_ready(1'b0),
      .io_ddr_mem_axi_ar_bits_addr(),
      .io_ddr_mem_axi_ar_bits_prot(),
      .io_ddr_mem_axi_ar_bits_id(),
      .io_ddr_mem_axi_ar_bits_len(),
      .io_ddr_mem_axi_ar_bits_size(),
      .io_ddr_mem_axi_ar_bits_burst(),
      .io_ddr_mem_axi_ar_bits_lock(),
      .io_ddr_mem_axi_ar_bits_cache(),
      .io_ddr_mem_axi_ar_bits_qos(),
      .io_ddr_mem_axi_ar_bits_region(),
      .io_ddr_mem_axi_r_valid(1'b0),
      .io_ddr_mem_axi_r_ready(),
      .io_ddr_mem_axi_r_bits_data(32'b0),
      .io_ddr_mem_axi_r_bits_id(6'b0),
      .io_ddr_mem_axi_r_bits_resp(2'b0),
      .io_ddr_mem_axi_r_bits_last(1'b0),
      .io_dm_req_valid(dm_req_valid),
      .io_dm_req_ready(dm_req_ready),
      .io_dm_req_bits_address(dm_req.addr),
      .io_dm_req_bits_data(dm_req.data),
      .io_dm_req_bits_op(dm_req.op),
      .io_dm_rsp_ready(dm_rsp_ready),
      .io_dm_rsp_valid(dm_rsp_valid),
      .io_dm_rsp_bits_data(dm_rsp.data),
      .io_dm_rsp_bits_op(dm_rsp.resp)
    );
endmodule
