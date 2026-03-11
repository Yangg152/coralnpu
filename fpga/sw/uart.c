// Copyright 2026 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "fpga/sw/uart.h"

#define UART1_BASE 0x40010000
#define UART_CTRL_OFFSET 0x10
#define UART_STATUS_OFFSET 0x14
#define UART_WDATA_OFFSET 0x1c

#define REG32(addr) (*(volatile uint32_t*)(addr))

void uart_init(uint32_t clock_frequency_mhz) {
  const uint64_t uart_ctrl_nco =
      ((uint64_t)115200 << 20) / (clock_frequency_mhz * 1000000);
  REG32(UART1_BASE + UART_CTRL_OFFSET) = (uint32_t)((uart_ctrl_nco << 16) | 3);
}

void uart_putc(char c) {
  volatile uint32_t* uart_status =
      (volatile uint32_t*)(UART1_BASE + UART_STATUS_OFFSET);
  volatile uint32_t* uart_wdata =
      (volatile uint32_t*)(UART1_BASE + UART_WDATA_OFFSET);
  while (*uart_status & 1) asm volatile("nop");
  *uart_wdata = (uint32_t)c;
}

void uart_puts(const char* s) {
  while (*s) uart_putc(*s++);
}

void uart_puthex8(uint8_t v) {
  const char* hex = "0123456789ABCDEF";
  uart_putc(hex[(v >> 4) & 0xF]);
  uart_putc(hex[v & 0xF]);
}

void uart_puthex32(uint32_t v) {
  const char* hex = "0123456789ABCDEF";
  for (int i = 7; i >= 0; i--) {
    uart_putc(hex[(v >> (i * 4)) & 0xF]);
  }
}
