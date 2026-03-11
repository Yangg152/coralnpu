# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Clock Signal
create_clock -period 10.00 -name sys_clk_pin -waveform {0 5} [get_ports clk_p_i]
set_property -dict { PACKAGE_PIN U13 IOSTANDARD DIFF_SSTL18_I } [get_ports { clk_p_i }];
set_property -dict { PACKAGE_PIN T13 IOSTANDARD DIFF_SSTL18_I } [get_ports { clk_n_i }];
create_clock -period 6.4 -name c0_sys_clk_p [get_ports c0_sys_clk_p]

# Generated Clocks
create_generated_clock -name clk_main [get_pin i_clkgen/i_clkgen/pll/CLKOUT0]
create_generated_clock -name clk_48MHz [get_pin i_clkgen/i_clkgen/pll/CLKOUT1]
create_generated_clock -name clk_aon [get_pin i_clkgen/i_clkgen/pll/CLKOUT4]

# Reset
set_property -dict { PACKAGE_PIN AR19 IOSTANDARD LVCMOS18 } [get_ports { rst_ni }];

# JTAG
# 500 kHz clock constraint
create_clock -period 2000.00 -name jtag_tck_i -waveform {0 1000} [get_ports {tck_i}]
set_property -dict { PACKAGE_PIN BE18 IOSTANDARD LVCMOS18 PULLTYPE PULLDOWN } [get_ports {tms_i}]
set_property -dict { PACKAGE_PIN BE17 IOSTANDARD LVCMOS18 PULLTYPE PULLDOWN } [get_ports {td_o}]
set_property -dict { PACKAGE_PIN BB19 IOSTANDARD LVCMOS18 PULLTYPE PULLDOWN } [get_ports {td_i}]
set_property -dict { PACKAGE_PIN AW18 IOSTANDARD LVCMOS18 PULLTYPE PULLDOWN } [get_ports {tck_i}]
set_property -dict { PACKAGE_PIN BC19 IOSTANDARD LVCMOS18 } [get_ports {trst_ni}]

# SPI
create_clock -period 83.333 -name spi_clk_i -waveform {0 41.667} [get_ports spi_clk_i]
set_property -dict { PACKAGE_PIN AV19 IOSTANDARD LVCMOS18 } [get_ports { spi_clk_i }];
set_property -dict { PACKAGE_PIN AW20 IOSTANDARD LVCMOS18 } [get_ports { spi_csb_i }];
set_property -dict { PACKAGE_PIN AV20 IOSTANDARD LVCMOS18 } [get_ports { spi_mosi_i }];
set_property -dict { PACKAGE_PIN AV18 IOSTANDARD LVCMOS18 } [get_ports { spi_miso_o }];

# UART0
set_property -dict { PACKAGE_PIN BF20 IOSTANDARD LVCMOS18 } [get_ports { uart_tx_o[0] }];
set_property -dict { PACKAGE_PIN BD20 IOSTANDARD LVCMOS18 } [get_ports { uart_rx_i[0] }];

# UART1
set_property -dict { PACKAGE_PIN R23 IOSTANDARD LVCMOS18 } [get_ports { uart_tx_o[1] }];
set_property -dict { PACKAGE_PIN T23 IOSTANDARD LVCMOS18 } [get_ports { uart_rx_i[1] }];

# LEDs
set_property -dict { PACKAGE_PIN T31 DRIVE 8 IOSTANDARD LVCMOS12 } [get_ports { io_halted }];
set_property -dict { PACKAGE_PIN P31 DRIVE 8 IOSTANDARD LVCMOS12 } [get_ports { io_fault }];
set_property -dict { PACKAGE_PIN N37 DRIVE 8 IOSTANDARD LVCMOS12 } [get_ports { ddr_cal_complete_o }];
set_property -dict { PACKAGE_PIN M38 DRIVE 8 IOSTANDARD LVCMOS12 } [get_ports { io_ddr_mem_axi_aw_ready }];
set_property -dict { PACKAGE_PIN L38 DRIVE 8 IOSTANDARD LVCMOS12 } [get_ports { io_ddr_mem_axi_ar_ready }];
set_property -dict { PACKAGE_PIN L36 DRIVE 8 IOSTANDARD LVCMOS12 } [get_ports { ddr_ui_clk }];
set_property -dict { PACKAGE_PIN K36 DRIVE 8 IOSTANDARD LVCMOS12 } [get_ports { ddr_ui_clk_sync_rst }];

# Asynchronous Clock Groups
# Define all primary, asynchronous clocks
set_clock_groups -asynchronous \
  -group [get_clocks -include_generated_clocks sys_clk_pin] \
  -group [get_clocks -include_generated_clocks c0_sys_clk_p] \
  -group [get_clocks spi_clk_i] \
  -group [get_clocks jtag_tck_i]

# SPI Probe Outputs (PMOD3) -> Reassigned to SpiMaster
# PMOD4: 1=AY38, 2=BA39, 3=AW35, 4=AY35, 7=AY40, 8=BA40, 9=AW36, 10=BC40
set_property -dict { PACKAGE_PIN AY38 IOSTANDARD LVCMOS18 } [get_ports { spim_mosi_o }]; # PMOD4_1 (D0)
set_property -dict { PACKAGE_PIN BA39 IOSTANDARD LVCMOS18 } [get_ports { spim_miso_i }]; # PMOD4_2 (D1)
set_property -dict { PACKAGE_PIN AW35 IOSTANDARD LVCMOS18 } [get_ports { gpio[0] }];     # PMOD4_3 (D2)
set_property -dict { PACKAGE_PIN AY35 IOSTANDARD LVCMOS18 } [get_ports { gpio[1] }];     # PMOD4_3 (D3)

# I2C (PMOD2)
set_property -dict { PACKAGE_PIN AR35 IOSTANDARD LVCMOS18 } [get_ports { i2c_scl }];     # PMOD2_9
set_property -dict { PACKAGE_PIN AT35 IOSTANDARD LVCMOS18 } [get_ports { i2c_sda }];     # PMOD2_3

set_property -dict { PACKAGE_PIN AY40 IOSTANDARD LVCMOS18 } [get_ports { spim_sclk_o }]; # PMOD4_7 (CLK)
set_property -dict { PACKAGE_PIN BA40 IOSTANDARD LVCMOS18 } [get_ports { spim_csb_o }];  # PMOD4_8 (CS)
set_property -dict { PACKAGE_PIN AW36 IOSTANDARD LVCMOS18 } [get_ports { gpio[2] }];     # PMOD4_9
set_property -dict { PACKAGE_PIN BC40 IOSTANDARD LVCMOS18 } [get_ports { gpio[3] }];     # PMOD4_10

set_property -dict { PACKAGE_PIN AU40 IOSTANDARD LVCMOS18 } [get_ports { spi_clk_probe_o }];
set_property -dict { PACKAGE_PIN AV40 IOSTANDARD LVCMOS18 } [get_ports { spi_csb_probe_o }];
set_property -dict { PACKAGE_PIN AW40 IOSTANDARD LVCMOS18 } [get_ports { spi_mosi_probe_o }];
set_property -dict { PACKAGE_PIN AY39 IOSTANDARD LVCMOS18 } [get_ports { spi_miso_probe_o }];

