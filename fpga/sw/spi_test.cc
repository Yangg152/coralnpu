// Copyright 2026 Google LLC
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

#include <stdint.h>

#define SPI_MASTER_BASE 0x40020000

#define SPI_REG_STATUS (SPI_MASTER_BASE + 0x00)
#define SPI_REG_CONTROL (SPI_MASTER_BASE + 0x04)
#define SPI_REG_TXDATA (SPI_MASTER_BASE + 0x08)
#define SPI_REG_RXDATA (SPI_MASTER_BASE + 0x0c)
#define SPI_REG_CSID (SPI_MASTER_BASE + 0x10)
#define SPI_REG_CSMODE (SPI_MASTER_BASE + 0x14)

#define REG32(addr) (*(volatile uint32_t*)(addr))

#include "fpga/sw/uart.h"

int main() {
  uart_init(CLOCK_FREQUENCY_MHZ);

  // 1. Enable SPI Master
  // Div = 20, CPOL=0, CPHA=0, Enable=1
  // Control register: Div(15:8), CPHA(2), CPOL(1), Enable(0)
  // 0x1401 -> Div=20, Enable=1
  REG32(SPI_REG_CONTROL) = 0x1401;

  // 2. Select CSID 0 and Auto Mode
  REG32(SPI_REG_CSID) = 0;
  REG32(SPI_REG_CSMODE) = 0;

  // 3. Send Data
  REG32(SPI_REG_TXDATA) = 0xDE;
  REG32(SPI_REG_TXDATA) = 0xAD;
  REG32(SPI_REG_TXDATA) = 0xBE;
  REG32(SPI_REG_TXDATA) = 0xEF;

  // Wait for TX FIFO to empty (Status bit 2 is TX Full, bit 0 is Busy)
  while (REG32(SPI_REG_STATUS) & 1);

  uart_puts("SPI transmit complete!\n");

  // Signal completion (using a known address or just loop)
  return 0;
}
