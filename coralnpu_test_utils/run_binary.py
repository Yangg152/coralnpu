#!/usr/bin/env python3
# /// script
# dependencies = [
#   "libusb_package",
#   "pyelftools",
#   "pyftdi",
# ]
# ///

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
import sys
import time

# To support 'import coralnpu_hw.coralnpu_test_utils' without Bazel:
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    import coralnpu_hw
except ImportError:
    import types
    _coralnpu_hw = types.ModuleType("coralnpu_hw")
    _coralnpu_hw.__path__ = [_project_root]
    sys.modules["coralnpu_hw"] = _coralnpu_hw

from elftools.elf.elffile import ELFFile
from coralnpu_hw.coralnpu_test_utils.ftdi_spi_master import FtdiSpiMaster


class BinaryRunner:
    """Loads and runs a binary on the CoralNPU hardware without input/output handling."""

    def __init__(
        self,
        elf_path,
        usb_serial,
        ftdi_port=1,
        csr_base_addr=0x30000,
        verify=False,
        exit_after_start=False,
    ):
        """
        Initializes the BinaryRunner.

        Args:
            elf_path: Path to the ELF file.
            usb_serial: USB serial number of the FTDI device.
            ftdi_port: Port number of the FTDI device.
            csr_base_addr: Base address for CSR registers.
            verify: Whether to verify the load by reading back memory.
            exit_after_start: Whether to exit immediately after starting the core.
        """
        self.elf_path = elf_path
        self.spi_master = FtdiSpiMaster(usb_serial, ftdi_port, csr_base_addr)
        self.entry_point = None
        self.verify = verify
        self.exit_after_start = exit_after_start
        self.status_msg_addr = None
        self.status_msg_size = 0
        self._parse_elf()

    def _parse_elf(self):
        """Parses the ELF file to find the entry point."""
        print(f"Parsing ELF file: {self.elf_path}")
        with open(self.elf_path, "rb") as f:
            elf = ELFFile(f)
            self.entry_point = elf.header["e_entry"]

            # Find inference_status_message symbol
            symtab = elf.get_section_by_name(".symtab")
            if symtab:
                syms = symtab.get_symbol_by_name("inference_status_message")
                if syms:
                    self.status_msg_addr = syms[0].entry["st_value"]
                    self.status_msg_size = syms[0].entry["st_size"]
                    print(
                        f"  Found 'inference_status_message' at 0x{self.status_msg_addr:x} (size {self.status_msg_size})"
                    )

        if self.entry_point is None:
            raise ValueError("Could not find entry point in ELF file.")
        print(f"  Found entry point at 0x{self.entry_point:x}")

    def _verify_load(self):
        """Verifies the ELF load by reading back memory and comparing."""
        print(f"Verifying ELF load: {self.elf_path}")
        with open(self.elf_path, "rb") as f:
            elf = ELFFile(f)
            for segment in elf.iter_segments(type="PT_LOAD"):
                paddr = segment.header.p_vaddr
                expected_data = segment.data()
                if not expected_data:
                    continue
                print(
                    f"  Verifying segment at 0x{paddr:x} ({len(expected_data)} bytes)..."
                )
                actual_data = self.spi_master.read_data(paddr, len(expected_data))
                if actual_data != expected_data:
                    # Find the first mismatch for better error reporting
                    for i in range(len(expected_data)):
                        if actual_data[i] != expected_data[i]:
                            raise ValueError(
                                f"Verification FAILED at address 0x{paddr + i:x}: "
                                f"expected 0x{expected_data[i]:02x}, "
                                f"got 0x{actual_data[i]:02x}"
                            )
        print("Verification SUCCESSFUL.")

    def run_binary(self):
        """Executes the binary load and run flow."""
        # TODO(atv): Re-enable this when toggling POR through FTDI doesn't break DDR.
        # self.spi_master.device_reset()
        self.spi_master.idle_clocking(20)

        # 1. Load ELF (without starting the core)
        print(f"Loading ELF file: {self.elf_path}")
        self.spi_master.load_elf(self.elf_path, start_core=False)

        # 1.5 Optional Verification
        if self.verify:
            self._verify_load()

        # 2. Set the entry point and start the core
        print(f"Setting entry point to 0x{self.entry_point:x} and starting core...")
        self.spi_master.set_entry_point(self.entry_point)
        self.spi_master.start_core()

        if self.exit_after_start:
            print("Exiting after start as requested.")
            return

        # 3. Wait for the core to halt with status polling
        print("Waiting for core to halt...")
        timeout = 60.0
        start_time = time.time()
        halt_addr = self.spi_master.csr_base_addr + 8
        last_status_msg = ""
        core_halted = False

        while time.time() - start_time < timeout:
            # Check halt status
            if self.spi_master.read_word(halt_addr) == 1:
                core_halted = True
                break

            # Poll status message if available
            if self.status_msg_addr:
                try:
                    # Read bytes and decode
                    status_bytes = self.spi_master.read_data(
                        self.status_msg_addr, self.status_msg_size, verbose=False
                    )
                    # Find null terminator or end of buffer
                    status_str = status_bytes.split(b"\0", 1)[0].decode(
                        "utf-8", errors="replace"
                    )
                    if status_str != last_status_msg:
                        print(f"Status: {status_str}")
                        last_status_msg = status_str
                        # Zero out the first byte of the status message to signal we've read it
                        self.spi_master.write_word(self.status_msg_addr, 0)
                except Exception as e:
                    print(f"Error reading status message: {e}")

            time.sleep(0.1)  # Poll interval

        if not core_halted:
            print("Binary execution FAILED: Core did not halt within timeout.")
        else:
            print("Binary execution COMPLETED: Core halted successfully.")
            # Print final status
            if self.status_msg_addr:
                try:
                    status_bytes = self.spi_master.read_data(
                        self.status_msg_addr, self.status_msg_size, verbose=False
                    )
                    status_str = status_bytes.split(b"\0", 1)[0].decode(
                        "utf-8", errors="replace"
                    )
                    if status_str != last_status_msg:
                        print(f"Final Status: {status_str}")
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description="Load and run a binary on CoralNPU.")
    parser.add_argument("elf_file", help="Path to the ELF file to run.")
    parser.add_argument(
        "--usb-serial", required=True, help="USB serial number of the FTDI device."
    )
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
        "--highmem",
        action="store_true",
        help="Use high memory (0x200000) for CSR base address.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the ELF load by reading back memory.",
    )
    parser.add_argument(
        "--exit-after-start",
        action="store_true",
        help="Exit immediately after starting the core.",
    )
    args = parser.parse_args()

    csr_base_addr = args.csr_base_addr
    if args.highmem:
        csr_base_addr = 0x200000

    try:
        runner = BinaryRunner(
            args.elf_file,
            args.usb_serial,
            args.ftdi_port,
            csr_base_addr,
            verify=args.verify,
            exit_after_start=args.exit_after_start,
        )
        runner.run_binary()
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
