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

#define GPIO_BASE 0x40030000

#define GPIO_DATA_IN (GPIO_BASE + 0x00)
#define GPIO_DATA_OUT (GPIO_BASE + 0x04)
#define GPIO_OUT_EN (GPIO_BASE + 0x08)

#define REG32(addr) (*(volatile uint32_t*)(addr))

#include "fpga/sw/uart.h"

int main() {
  uart_init(CLOCK_FREQUENCY_MHZ);

  // 1. Configure all pins as output
  REG32(GPIO_OUT_EN) = 0xFF;

  // 2. Write pattern 0xAA
  REG32(GPIO_DATA_OUT) = 0xAA;

  // 3. Read back from output register
  volatile uint32_t val = REG32(GPIO_DATA_OUT);
  if ((val & 0xFF) != 0xAA) {
    uart_puts("GPIO test FAIL!");
    return 1;  // Fail
  }

  // 4. Write pattern 0x55
  REG32(GPIO_DATA_OUT) = 0x55;

  // 5. Read back
  val = REG32(GPIO_DATA_OUT);
  if ((val & 0xFF) != 0x55) {
    uart_puts("GPIO test FAIL!");
    return 2;  // Fail
  }

  // 6. Test Input (Loopback)
  // The DPI model implements a loopback: if output enabled, input = output.
  // So reading DATA_IN should match DATA_OUT.
  val = REG32(GPIO_DATA_IN);
  if ((val & 0xFF) != 0x55) {
    uart_puts("GPIO test FAIL!");
    return 3;  // Fail: Loopback mismatch
  }

  uart_puts("GPIO test PASS!");

  return 0;  // Pass
}
