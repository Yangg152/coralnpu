// Trivial test to validate coralnpu_v2_sim_test infrastructure.
#include "fpga/sw/uart.h"

int main() {
  uart_init(CLOCK_FREQUENCY_MHZ);
  uart_puts("PASS\r\n");
  return 0;
}
