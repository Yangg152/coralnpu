#!/usr/bin/env python3
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import random
import sys
import time
import traceback
import numpy as np

try:
    from coralnpu_test_utils.ftdi_spi_master import FtdiSpiMaster
except ImportError:
    FtdiSpiMaster = None

from coralnpu_test_utils.spi_constants import SpiRegAddress, SpiCommand, TlStatus


class SimSpiMaster:
    """A simulation-compatible SPI master using the TCP DPI interface."""

    def __init__(self, port=5555):
        # Try to import SPIDriver from the utils directory
        # Adjust path assuming this script is in coralnpu_test_utils/
        driver_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../utils/coralnpu_soc_loader")
        )
        if driver_path not in sys.path:
            sys.path.append(driver_path)

        try:
            from spi_driver import SPIDriver
        except ImportError:
            # Fallback for bazel run where imports might be different
            try:
                from utils.coralnpu_soc_loader.spi_driver import SPIDriver
            except ImportError:
                raise ImportError(
                    f"Could not import SPIDriver. Checked {driver_path} and utils.coralnpu_soc_loader.spi_driver"
                )

        print(f"Attempting to connect to simulator on port {port}...")
        for i in range(30):
            try:
                self.driver = SPIDriver(port)
                print(f"Connected to simulator on port {port}")
                break
            except (ConnectionRefusedError, OSError) as e:
                if i % 5 == 0:
                    print(f"Waiting for simulator... ({e})")
                time.sleep(1)
        else:
            raise ConnectionRefusedError(
                f"Could not connect to simulator on port {port} after 30 seconds."
            )

    def close(self):
        if hasattr(self, "driver") and self.driver:
            self.driver.close()

    def idle_clocking(self, cycles):
        self.driver.idle_clocking(cycles)

    def write_reg(self, addr, data):
        self.driver.write_reg(addr, data)

    def read_spi_domain_reg_16b(self, addr):
        return self.driver.read_spi_domain_reg_16b(addr)

    def poll_reg_for_value(self, addr, expected, timeout=1.0):
        # SPIDriver's poll is blocking on the server side with a count.
        # We'll use a reasonable count.
        return self.driver.poll_reg_for_value(addr, expected, max_polls=1000)

    def read_line(self, address):
        """Reads a single 128-bit line from memory via SPI."""
        # 1. Configure the read
        self.driver.write_reg(SpiRegAddress.TL_ADDR_REG_0, (address >> 0) & 0xFF)
        self.driver.write_reg(SpiRegAddress.TL_ADDR_REG_1, (address >> 8) & 0xFF)
        self.driver.write_reg(SpiRegAddress.TL_ADDR_REG_2, (address >> 16) & 0xFF)
        self.driver.write_reg(SpiRegAddress.TL_ADDR_REG_3, (address >> 24) & 0xFF)
        self.driver.write_reg_16b(SpiRegAddress.TL_LEN_REG_L, 0)  # 1 beat

        # 2. Issue the read command
        self.driver.write_reg(SpiRegAddress.TL_CMD_REG, SpiCommand.CMD_READ_START)

        # 3. Poll for completion
        if not self.driver.poll_reg_for_value(
            SpiRegAddress.TL_STATUS_REG, TlStatus.DONE
        ):
            raise RuntimeError(f"Timed out waiting for TL read at 0x{address:x}")

        # 4. Check status
        bytes_available = self.driver.read_spi_domain_reg_16b(
            SpiRegAddress.BULK_READ_STATUS_REG_L
        )
        if bytes_available != 16:
            raise RuntimeError(
                f"Expected 16 bytes, but status reported {bytes_available}"
            )

        # 5. Bulk read
        read_data_bytes = self.driver.bulk_read(bytes_available)
        read_data = int.from_bytes(bytes(read_data_bytes), "little")

        # 6. Clear command
        self.driver.write_reg(SpiRegAddress.TL_CMD_REG, SpiCommand.CMD_NULL)

        return read_data

    def write_lines(self, start_addr, num_beats, data_as_bytes):
        """Writes a contiguous block of data."""
        # SPIDriver has a packed_write_transaction that handles this
        data_int = int.from_bytes(data_as_bytes, "little")
        # Note: SPIDriver.packed_write_transaction takes data as int if using older version?
        # Let's check spi_driver.py again.
        # It takes `data` and does `payload = data.to_bytes(...)`. So yes, expects int.

        # However, `write_lines_via_spi` in loader.py does:
        # data_int = int.from_bytes(data_bytes, byteorder='little')
        # driver.packed_write_transaction(address, num_lines, data_int)

        self.driver.packed_write_transaction(start_addr, num_beats, data_int)

        # Poll for completion
        if not self.driver.poll_reg_for_value(
            SpiRegAddress.TL_WRITE_STATUS_REG, TlStatus.DONE, max_polls=2000
        ):
            raise RuntimeError(f"Timed out waiting for TL write at 0x{start_addr:x}")

        # Acknowledge
        self.driver.write_reg(SpiRegAddress.TL_CMD_REG, SpiCommand.CMD_NULL)

    def write_line(self, address, data):
        data_as_bytes = data.to_bytes(16, "little")
        self.write_lines(address, 1, data_as_bytes)

    def load_data(self, data, address):
        """Loads data handling unaligned start/end."""
        size = len(data)
        start_address = address
        end_address = start_address + size
        data_ptr = 0

        # 1. Handle unaligned start
        start_offset = start_address % 16
        if start_offset != 0:
            line_addr = start_address - start_offset
            bytes_to_write = min(16 - start_offset, size)

            data_chunk = data[data_ptr : data_ptr + bytes_to_write]
            data_ptr += bytes_to_write

            old_line_int = self.read_line(line_addr)
            old_line_bytes = old_line_int.to_bytes(16, "little")

            new_line_bytes = bytearray(old_line_bytes)
            new_line_bytes[start_offset : start_offset + bytes_to_write] = data_chunk
            new_line_int = int.from_bytes(new_line_bytes, "little")

            self.write_line(line_addr, new_line_int)

        # 2. Handle aligned middle
        loop_start_addr = (start_address + 15) & ~0xF
        loop_end_addr = end_address & ~0xF
        if loop_end_addr > loop_start_addr:
            full_lines_data_size = loop_end_addr - loop_start_addr

            # Chunking 4096 bytes
            for i in range(0, full_lines_data_size, 4096):
                chunk_start_addr = loop_start_addr + i
                chunk_size = min(4096, full_lines_data_size - i)
                num_lines = chunk_size // 16

                data_chunk_bytes = data[data_ptr : data_ptr + chunk_size]
                data_ptr += chunk_size

                self.write_lines(chunk_start_addr, num_lines, data_chunk_bytes)

        # 3. Handle unaligned end
        end_offset = end_address % 16
        if end_offset != 0:
            line_addr = end_address - end_offset
            # Avoid re-writing start line if it was already handled in Step 1
            # (Step 1 handles the line if start_offset != 0)
            if start_offset == 0 or line_addr != (start_address - start_offset):
                bytes_to_write = end_offset
                data_chunk = data[data_ptr : data_ptr + bytes_to_write]

                old_line_int = self.read_line(line_addr)
                old_line_bytes = old_line_int.to_bytes(16, "little")

                new_line_bytes = bytearray(old_line_bytes)
                new_line_bytes[0:bytes_to_write] = data_chunk
                new_line_int = int.from_bytes(new_line_bytes, "little")

                self.write_line(line_addr, new_line_int)

    def read_data(self, address, size, verbose=False):
        if size == 0:
            return bytearray()

        data = bytearray()
        bytes_remaining = size
        current_addr = address

        # 1. Unaligned start
        start_offset = current_addr % 16
        if start_offset != 0:
            line_addr = current_addr - start_offset
            bytes_to_read = min(16 - start_offset, bytes_remaining)
            line_data = self.read_line(line_addr)
            line_bytes = line_data.to_bytes(16, "little")
            data.extend(line_bytes[start_offset : start_offset + bytes_to_read])
            bytes_remaining -= bytes_to_read
            current_addr += bytes_to_read

        # 2. Aligned chunks
        while bytes_remaining > 0:
            # TL transaction size limit (2kB)
            tl_txn_size = min(2048, bytes_remaining)
            num_beats = (tl_txn_size + 15) // 16
            expected_bytes = num_beats * 16

            # Configure read
            num_beats_val = num_beats - 1
            # We don't have packed write helper for the setup command buffer like FtdiSpiMaster does.
            # We have to issue writes to registers.
            self.driver.write_reg(
                SpiRegAddress.TL_ADDR_REG_0, (current_addr >> 0) & 0xFF
            )
            self.driver.write_reg(
                SpiRegAddress.TL_ADDR_REG_1, (current_addr >> 8) & 0xFF
            )
            self.driver.write_reg(
                SpiRegAddress.TL_ADDR_REG_2, (current_addr >> 16) & 0xFF
            )
            self.driver.write_reg(
                SpiRegAddress.TL_ADDR_REG_3, (current_addr >> 24) & 0xFF
            )
            self.driver.write_reg(SpiRegAddress.TL_LEN_REG_L, num_beats_val & 0xFF)
            self.driver.write_reg(
                SpiRegAddress.TL_LEN_REG_H, (num_beats_val >> 8) & 0xFF
            )

            self.driver.write_reg(SpiRegAddress.TL_CMD_REG, SpiCommand.CMD_READ_START)

            if not self.driver.poll_reg_for_value(
                SpiRegAddress.TL_STATUS_REG, TlStatus.DONE
            ):
                raise RuntimeError(
                    f"Timed out waiting for bulk TL read at 0x{current_addr:x}"
                )

            # Check available bytes
            # Note: polling loop logic in driver
            bytes_available = self.driver.read_spi_domain_reg_16b(
                SpiRegAddress.BULK_READ_STATUS_REG_L
            )
            # Retry a few times if 0? FtdiSpiMaster polls this.
            # SPIDriver.read_spi_domain_reg_16b is a single transaction.
            # We might need to poll it.
            for _ in range(100):
                if bytes_available == expected_bytes:
                    break
                time.sleep(0.01)
                bytes_available = self.driver.read_spi_domain_reg_16b(
                    SpiRegAddress.BULK_READ_STATUS_REG_L
                )

            if bytes_available != expected_bytes:
                raise RuntimeError(
                    f"Timed out waiting for {expected_bytes} bytes at 0x{current_addr:x}, got {bytes_available}"
                )

            # Bulk read
            chunk_data = self.driver.bulk_read(expected_bytes)
            data.extend(chunk_data)

            # Ack
            self.driver.write_reg(SpiRegAddress.TL_CMD_REG, SpiCommand.CMD_NULL)

            bytes_remaining -= expected_bytes
            current_addr += expected_bytes

        return data[:size]


class SramTestRunner:
    """Runs a SRAM test on the CoralNPU hardware."""

    SRAM_ADDR = 0x20000000

    def __init__(
        self,
        usb_serial,
        ftdi_port=1,
        csr_base_addr=0x30000,
        continue_on_error=False,
        simulation=False,
        sim_port=5555,
        max_size=256,
    ):
        self.MAX_SIZE = max_size
        """
        Initializes the SramTestRunner.

        Args:
            usb_serial: USB serial number of the FTDI device.
            ftdi_port: Port number of the FTDI device.
            continue_on_error: Whether to continue testing after a failure.
        """
        if simulation:
            print(f"Initializing Simulation SPI Master on port {sim_port}")
            self.spi_master = SimSpiMaster(sim_port)
        else:
            if FtdiSpiMaster is None:
                raise ImportError(
                    "FtdiSpiMaster could not be imported. Is pyftdi installed?"
                )
            self.spi_master = FtdiSpiMaster(usb_serial, ftdi_port, csr_base_addr)

        self.continue_on_error = continue_on_error
        self.results = []

    def _generate_pattern_data(self, size, pattern_type):
        """Generates data based on the requested pattern type."""
        if pattern_type == "zeros":
            return np.zeros(size, dtype=np.uint8)
        elif pattern_type == "ones":
            return np.full(size, 0xFF, dtype=np.uint8)
        elif pattern_type == "0x55":
            return np.full(size, 0x55, dtype=np.uint8)
        elif pattern_type == "0xAA":
            return np.full(size, 0xAA, dtype=np.uint8)
        elif pattern_type == "incrementing":
            return np.arange(size, dtype=np.uint8)
        elif pattern_type == "random":
            return np.random.randint(0, 256, size=size, dtype=np.uint8)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

    def run_test(self):
        """Executes the full SRAM test flow."""
        try:
            self.spi_master.idle_clocking(20)
            self.results = []

            patterns = ["zeros", "ones", "0x55", "0xAA", "incrementing", "random"]

            # Power of two sizes from 1 up to MAX_SIZE
            sizes = []
            curr_size = 1
            while curr_size <= self.MAX_SIZE:
                sizes.append(curr_size)
                curr_size *= 2

            print(f"Starting SRAM Test Suite.")
            print(f"Target Address: 0x{self.SRAM_ADDR:x}")
            print(f"Max Size: {self.MAX_SIZE} bytes")
            if getattr(self, "single_size", None):
                print(f"Filtering: Size={self.single_size}")
                sizes = [self.single_size]
            if getattr(self, "single_pattern", None):
                print(f"Filtering: Pattern={self.single_pattern}")
                patterns = [self.single_pattern]

            print("-" * 40)

            for size in sizes:
                for pattern in patterns:
                    try:
                        success, message = self._run_single_test(size, pattern)
                        self.results.append(
                            {
                                "size": size,
                                "pattern": pattern,
                                "success": success,
                                "message": message,
                            }
                        )

                        if not success and not self.continue_on_error:
                            self._print_summary()
                            return  # Stop on first failure
                    except Exception as e:
                        self.results.append(
                            {
                                "size": size,
                                "pattern": pattern,
                                "success": False,
                                "message": f"Exception: {str(e)}",
                            }
                        )
                        if not self.continue_on_error:
                            self._print_summary()
                            return

            print("-" * 40)
            self._print_summary()
        finally:
            if hasattr(self, "spi_master") and self.spi_master:
                self.spi_master.close()

    def _print_summary(self):
        """Prints a summary of test results."""
        print("\nTest Summary:")
        print(f"{'Size':<10} | {'Pattern':<10} | {'Result':<10} | {'Details'}")
        print("-" * 80)

        failures = 0
        for r in self.results:
            status = "PASS" if r["success"] else "FAIL"
            if not r["success"]:
                failures += 1
            print(
                f"{r['size']:<10} | {r['pattern']:<10} | {status:<10} | {r['message']}"
            )

        print("-" * 80)
        print(f"Total Tests: {len(self.results)}")
        print(f"Passed:      {len(self.results) - failures}")
        print(f"Failed:      {failures}")

    def _get_mismatch_ranges(self, indices):
        """Groups a list of sorted indices into contiguous ranges."""
        if len(indices) == 0:
            return []

        ranges = []
        start = indices[0]
        prev = start

        for curr in indices[1:]:
            if curr != prev + 1:
                ranges.append((start, prev))
                start = curr
            prev = curr
        ranges.append((start, prev))
        return ranges

    def _format_ranges(self, ranges, limit=None):
        """Formats range tuples into a string string, with optional limiting."""
        formatted = []
        for start, end in ranges:
            if start == end:
                formatted.append(f"0x{start:x}")
            else:
                formatted.append(f"0x{start:x}-0x{end:x}")

        if limit and len(formatted) > limit:
            return (
                ", ".join(formatted[:limit]) + f", ... ({len(formatted)} total ranges)"
            )
        return ", ".join(formatted)

    def _run_single_test(self, size, pattern):
        """Runs a single read/write verification test."""
        print(f"Running test: size={size}, pattern={pattern}")
        golden_data = self._generate_pattern_data(size, pattern)

        # 1. Load data
        print(f"(Loading {size} bytes to 0x{self.SRAM_ADDR:x}...)")
        self.spi_master.load_data(golden_data.tobytes(), self.SRAM_ADDR)

        # 2. Read back
        print(f"(Reading back {size} bytes from 0x{self.SRAM_ADDR:x}...)")
        result_data = self.spi_master.read_data(self.SRAM_ADDR, size, verbose=False)
        result_array = np.frombuffer(result_data, dtype=np.uint8)

        # 3. Verify
        print("(Verifying...)")
        if not np.array_equal(golden_data, result_array):
            mismatch_indices = np.where(golden_data != result_array)[0]
            count = len(mismatch_indices)
            ranges = self._get_mismatch_ranges(mismatch_indices)
            print(f"FAILED: {count} errors found.")
            # Print up to 16 errors for debugging
            for idx in mismatch_indices[:16]:
                print(
                    f"  Addr 0x{self.SRAM_ADDR + idx:x}: Expected 0x{golden_data[idx]:02x}, Got 0x{result_array[idx]:02x}"
                )

            return False, f"{count} errs. Ranges: {self._format_ranges(ranges)}"

        print("PASSED")
        return True, "OK"


def main():
    # Force unbuffered stdout
    sys.stdout.reconfigure(line_buffering=True)
    print("SRAM Test Script Started.", flush=True)

    parser = argparse.ArgumentParser(description="Run SRAM test on CoralNPU.")
    parser.add_argument("--usb-serial", help="USB serial number of the FTDI device.")
    parser.add_argument(
        "--ftdi-port", type=int, default=1, help="Port number of the FTDI device."
    )
    parser.add_argument(
        "--csr-base-addr",
        type=lambda x: int(x, 0),
        default=0x30000,
        help="Base address for CSR registers (can be hex, default: 0x30000).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue testing even if a test fails.",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode (connect to TCP port).",
    )
    parser.add_argument(
        "--sim-port", type=int, default=5555, help="TCP port for simulation mode."
    )
    parser.add_argument(
        "--max-size", type=int, default=256, help="Maximum test size in bytes."
    )
    parser.add_argument(
        "--single-test", help="Run only a specific test pattern (e.g., 'incrementing')."
    )
    parser.add_argument(
        "--single-size", type=int, help="Run only a specific test size."
    )
    args = parser.parse_args()

    if not args.simulation and not args.usb_serial:
        parser.error("--usb-serial is required unless --simulation is used.")

    try:
        runner = SramTestRunner(
            args.usb_serial,
            args.ftdi_port,
            args.csr_base_addr,
            continue_on_error=args.continue_on_error,
            simulation=args.simulation,
            sim_port=args.sim_port,
            max_size=args.max_size,
        )

        runner.single_size = None
        if args.single_size:
            # Monkey patch run_test method or just use internal logic?
            # Cleaner to pass these as args to run_test if I were refactoring,
            # but I'll attach them to the runner instance for minimal intrusion
            runner.single_size = args.single_size

        runner.single_pattern = None
        if args.single_test:
            runner.single_pattern = args.single_test

        runner.run_test()
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
