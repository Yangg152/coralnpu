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

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, with_timeout

from coralnpu_test_utils.TileLinkULInterface import TileLinkULInterface, create_a_channel_req
from coralnpu_test_utils.secded_golden import get_cmd_intg, get_data_intg, get_rsp_intg

# --- Configuration Constants ---
# These constants are derived from CrossbarConfig.scala to make tests readable.
HOST_MAP = {"coralnpu_core": 0, "spi2tlul": 1, "test_host_32": 2}
DEVICE_MAP = {
    "coralnpu_device": 0,
    "rom": 1,
    "sram": 2,
    "uart0": 3,
    "uart1": 4,
}
SRAM_BASE = 0x20000000
UART1_BASE = 0x40010000
CORALNPU_DEVICE_BASE = 0x00000000
INVALID_ADDR = 0x50000000
TIMEOUT_CYCLES = 500


# --- Test Setup ---
async def setup_dut(dut):
    """Common setup logic for all tests."""
    # Start the main clock
    clock = Clock(dut.clock, 6)
    cocotb.start_soon(clock.start())

    # Start the asynchronous test clock
    test_clock = Clock(dut.io_async_ports_hosts_test_clock, 10)
    cocotb.start_soon(test_clock.start())

    # Create a dictionary of TileLink interfaces for all hosts and devices
    host_widths = {"coralnpu_core": 128, "spi2tlul": 128, "test_host_32": 32}
    device_widths = {
        "coralnpu_device": 128,
        "rom": 32,
        "sram": 32,
        "uart0": 32,
        "uart1": 32,
    }

    interfaces = {
        "hosts": [
            TileLinkULInterface(dut,
                                host_if_name=f"io_hosts_{name}",
                                clock_name="clock" if name != "test_host_32" else "io_async_ports_hosts_test_clock",
                                reset_name="reset" if name != "test_host_32" else "io_async_ports_hosts_test_reset",
                                width=host_widths[name])
            for name, _ in HOST_MAP.items()
        ],
        "devices": [
            TileLinkULInterface(dut,
                                device_if_name=f"io_devices_{name}",
                                clock_name="clock",
                                reset_name="reset",
                                width=device_widths[name])
            for name, _ in DEVICE_MAP.items()
        ],
    }

    # Reset the DUT
    dut.reset.value = 1
    dut.io_async_ports_hosts_test_reset.value = 1
    await ClockCycles(dut.clock, 5)
    dut.reset.value = 0
    dut.io_async_ports_hosts_test_reset.value = 0
    await ClockCycles(dut.clock, 5)

    return interfaces, clock



# --- Test Cases ---


@cocotb.test(timeout_time=10, timeout_unit="us")
async def test_coralnpu_core_to_sram(dut):
    """Verify a simple write/read transaction from coralnpu_core to sram."""
    interfaces, clock = await setup_dut(dut)
    host_if = interfaces["hosts"][HOST_MAP["coralnpu_core"]]
    device_if = interfaces["devices"][DEVICE_MAP["sram"]]
    timeout_ns = TIMEOUT_CYCLES * clock.period

    # Send a 128-bit write request from the host
    test_data = 0x112233445566778899AABBCCDDEEFF00
    write_txn = create_a_channel_req(address=SRAM_BASE,
                                     data=test_data,
                                     mask=0xFFFF,
                                     width=host_if.width)
    await with_timeout(host_if.host_put(write_txn), timeout_ns, "ns")

    # Expect four 32-bit transactions on the device side, order not guaranteed
    received_reqs = []
    for _ in range(4):
        req = await with_timeout(device_if.device_get_request(), timeout_ns,
                                 "ns")
        received_reqs.append(req)
        await with_timeout(
            device_if.device_respond(opcode=0,
                                     param=0,
                                     size=req["size"],
                                     source=req["source"]), timeout_ns, "ns")

    # Sort received requests by address for comparison
    received_reqs.sort(key=lambda r: r["address"].integer)

    # Verify all beats were received correctly
    for i in range(4):
        assert received_reqs[i]["address"] == SRAM_BASE + (i * 4)
        expected_data = (test_data >> (i * 32)) & 0xFFFFFFFF
        assert received_reqs[i]["data"] == expected_data

    # Use the last beat (highest address) for the response source
    last_req = received_reqs[-1]

    # Receive the response on the host side
    resp = await with_timeout(host_if.host_get_response(), timeout_ns, "ns")
    assert resp["error"] == 0
    assert resp["source"] == write_txn["source"]


@cocotb.test(timeout_time=10, timeout_unit="us")
async def test_coralnpu_core_to_invalid_addr(dut):
    """Verify that a request to an unmapped address gets an error response."""
    interfaces, clock = await setup_dut(dut)
    host_if = interfaces["hosts"][HOST_MAP["coralnpu_core"]]
    timeout_ns = TIMEOUT_CYCLES * clock.period

    # Send a write request to an invalid address
    write_txn = create_a_channel_req(address=INVALID_ADDR,
                                     data=0,
                                     mask=0xF,
                                     width=host_if.width)
    await with_timeout(host_if.host_put(write_txn), timeout_ns, "ns")

    # Expect an error response
    try:
        resp = await with_timeout(host_if.host_get_response(), timeout_ns,
                                  "ns")
        assert resp["error"] == 1
        assert resp["source"] == write_txn["source"]
    except Exception as e:
        # Allow the simulation to run for a few more cycles to get a clean waveform
        await ClockCycles(dut.clock, 20)
        raise e

@cocotb.test(timeout_time=10, timeout_unit="us")
async def test_coralnpu_core_to_uart1(dut):
    """Verify a 128-bit to 32-bit write transaction."""
    interfaces, clock = await setup_dut(dut)
    host_if = interfaces["hosts"][HOST_MAP["coralnpu_core"]]
    device_if = interfaces["devices"][DEVICE_MAP["uart1"]]
    timeout_ns = TIMEOUT_CYCLES * clock.period

    # Send a 128-bit write request
    test_data = 0x112233445566778899AABBCCDDEEFF00
    write_txn = create_a_channel_req(address=UART1_BASE,
                                     data=test_data,
                                     mask=0xF0F0,
                                     width=host_if.width)
    await with_timeout(host_if.host_put(write_txn), timeout_ns, "ns")

    # Expect four 32-bit transactions on the device side, order not guaranteed
    received_reqs = []
    for i in range(4):
        req = await with_timeout(device_if.device_get_request(), timeout_ns,
                                 "ns")
        received_reqs.append(req)
        await with_timeout(
            device_if.device_respond(opcode=0,
                                     param=0,
                                     size=req["size"],
                                     source=req["source"],
                                     width=device_if.width), timeout_ns, "ns")

    # Sort received requests by address for comparison
    received_reqs.sort(key=lambda r: r["address"].integer)

    # Verify all beats were received correctly
    for i in range(4):
        assert received_reqs[i]["address"] == UART1_BASE + (i * 4)
        expected_data = (test_data >> (i * 32)) & 0xFFFFFFFF
        assert received_reqs[i]["data"] == expected_data
        expected_mask = 0xF if i in [1, 3] else 0x0
        assert received_reqs[i]["mask"] == expected_mask

    # Use the last beat (highest address) for the response source
    last_req = received_reqs[-1]

    # Receive the response on the host side
    resp = await with_timeout(host_if.host_get_response(), timeout_ns, "ns")
    assert resp["error"] == 0
    assert resp["source"] == write_txn["source"]


@cocotb.test(timeout_time=10, timeout_unit="us")
async def test_test_host_32_to_coralnpu_device(dut):
    """Verify a 32-bit to 128-bit write transaction."""
    interfaces, clock = await setup_dut(dut)
    host_if = interfaces["hosts"][HOST_MAP["test_host_32"]]
    device_if = interfaces["devices"][DEVICE_MAP["coralnpu_device"]]
    timeout_ns = TIMEOUT_CYCLES * clock.period

    # Send a 32-bit write request
    write_txn = create_a_channel_req(address=CORALNPU_DEVICE_BASE,
                                     data=0x12345678,
                                     mask=0xF,
                                     width=host_if.width)
    await with_timeout(host_if.host_put(write_txn), timeout_ns, "ns")

    # Expect a single 128-bit transaction on the device side
    req = await with_timeout(device_if.device_get_request(), timeout_ns, "ns")
    assert req["address"] == CORALNPU_DEVICE_BASE
    assert req["data"] == 0x12345678

    # Send a response from the device
    await with_timeout(
        device_if.device_respond(opcode=0,
                                 param=0,
                                 size=req["size"],
                                 source=req["source"],
                                 width=device_if.width), timeout_ns, "ns")

    # Expect a single response on the host side
    resp = await with_timeout(host_if.host_get_response(), timeout_ns, "ns")
    assert resp["error"] == 0


@cocotb.test(timeout_time=10, timeout_unit="us")
async def test_coralnpu_core_to_coralnpu_device(dut):
    """Verify a 128-bit to 128-bit write transaction (no bridge)."""
    interfaces, clock = await setup_dut(dut)
    host_if = interfaces["hosts"][HOST_MAP["coralnpu_core"]]
    device_if = interfaces["devices"][DEVICE_MAP["coralnpu_device"]]
    timeout_ns = TIMEOUT_CYCLES * clock.period

    # Send a 128-bit write request
    write_txn = create_a_channel_req(address=CORALNPU_DEVICE_BASE,
                                     data=0x112233445566778899AABBCCDDEEFF00,
                                     mask=0xFFFF,
                                     width=host_if.width)
    await with_timeout(host_if.host_put(write_txn), timeout_ns, "ns")

    # Expect a single 128-bit transaction on the device side
    req = await with_timeout(device_if.device_get_request(), timeout_ns, "ns")
    assert req["address"] == CORALNPU_DEVICE_BASE
    assert req["data"] == 0x112233445566778899AABBCCDDEEFF00

    # Send a response from the device
    await with_timeout(
        device_if.device_respond(opcode=0,
                                 param=0,
                                 size=req["size"],
                                 source=req["source"]), timeout_ns, "ns")

    # Expect a single response on the host side
    resp = await with_timeout(host_if.host_get_response(), timeout_ns, "ns")
    assert resp["error"] == 0





@cocotb.test(timeout_time=10, timeout_unit="us")
async def test_test_host_32_to_coralnpu_device_csr_read(dut):
    """Verify that test_host_32 can correctly read a CSR from the CoralNPU device.

    This test specifically checks the return path through the width bridge.
    """
    interfaces, clock = await setup_dut(dut)
    host_if = interfaces["hosts"][HOST_MAP["test_host_32"]]
    device_if = interfaces["devices"][DEVICE_MAP["coralnpu_device"]]
    timeout_ns = TIMEOUT_CYCLES * clock.period
    csr_addr = CORALNPU_DEVICE_BASE + 0x8  # Match the CSR address
    halted_status = 0x1  # Bit 0 for halted

    async def device_responder():
        """A mock responder for the coralnpu_device."""
        req = await with_timeout(device_if.device_get_request(), timeout_ns,
                                 "ns")
        # The address should be aligned to the device width (128-bit)
        aligned_addr = csr_addr
        assert req["address"] == aligned_addr, f"Expected aligned address 0x{aligned_addr:X}, but got 0x{req['address'].integer:X}"
        # The CSR data is in the third 32-bit lane of the 128-bit bus.
        resp_data = halted_status << 64
        await with_timeout(
            device_if.device_respond(
                opcode=1,  # AccessAckData
                param=0,
                size=req["size"],
                source=req["source"],
                data=resp_data,
                width=device_if.width,
            ),
            timeout_ns,
            "ns")

    # Start the device responder coroutine
    cocotb.start_soon(device_responder())

    # Send a 32-bit read request from the host
    # TODO(atv): Do this thru helper?
    read_txn = {
        "opcode": 4,  # Get
        "param": 0,
        "size": 2,  # 4 bytes
        "source": 1,
        "address": csr_addr,
        "mask": 0xF,
        "data": 0,
        "user": {
            "cmd_intg": 0,
            "data_intg": 0,
            "instr_type": 0,
            "rsvd": 0
        }
    }
    read_txn["user"]["cmd_intg"] = get_cmd_intg(read_txn, width=host_if.width)
    read_txn["user"]["data_intg"] = get_data_intg(read_txn["data"],
                                                  width=host_if.width)
    await with_timeout(host_if.host_put(read_txn), timeout_ns, "ns")

    # Expect a single response on the host side with the correct data
    resp = await with_timeout(host_if.host_get_response(), timeout_ns, "ns")
    assert resp["error"] == 0
    assert resp[
        "data"] == halted_status, f"Expected CSR data {halted_status}, but got {resp['data']}"


@cocotb.test(timeout_time=10, timeout_unit="us")
async def test_test_host_32_to_coralnpu_device_specific_addr(dut):
    """Verify a write to a specific address in the coralnpu_device range."""
    interfaces, clock = await setup_dut(dut)
    host_if = interfaces["hosts"][HOST_MAP["test_host_32"]]
    device_if = interfaces["devices"][DEVICE_MAP["coralnpu_device"]]
    timeout_ns = TIMEOUT_CYCLES * clock.period

    # Send a 32-bit write request to 0x30000
    test_addr = 0x30000
    write_txn = create_a_channel_req(address=test_addr,
                                     data=0xDEADBEEF,
                                     mask=0xF,
                                     width=host_if.width)
    await with_timeout(host_if.host_put(write_txn), timeout_ns, "ns")

    # Expect a single 128-bit transaction on the device side
    req = await with_timeout(device_if.device_get_request(), timeout_ns, "ns")
    assert req[
        "address"] == test_addr, f"Expected address 0x{test_addr:X}, but got 0x{req['address'].integer:X}"
    assert req["data"] == 0xDEADBEEF

    # Send a response from the device
    await with_timeout(
        device_if.device_respond(opcode=0,
                                 param=0,
                                 size=req["size"],
                                 source=req["source"],
                                 width=device_if.width), timeout_ns, "ns")

    # Expect a single response on the host side
    resp = await with_timeout(host_if.host_get_response(), timeout_ns, "ns")
    assert resp["error"] == 0


@cocotb.test(timeout_time=10, timeout_unit="us")
async def test_wide_to_narrow_integrity(dut):
    """Verify integrity is checked and regenerated across the width bridge."""
    interfaces, clock = await setup_dut(dut)
    host_if = interfaces["hosts"][HOST_MAP["coralnpu_core"]]
    device_if = interfaces["devices"][DEVICE_MAP["uart1"]]
    timeout_ns = TIMEOUT_CYCLES * clock.period

    # Send a 128-bit write request from the host with correct integrity
    test_data = 0x112233445566778899AABBCCDDEEFF00
    write_txn = create_a_channel_req(address=UART1_BASE,
                                     data=test_data,
                                     mask=0xFFFF,
                                     width=host_if.width)

    await with_timeout(host_if.host_put(write_txn), timeout_ns, "ns")

    # Expect four 32-bit transactions on the device side
    received_reqs = []
    for i in range(4):
        req = await with_timeout(device_if.device_get_request(), timeout_ns,
                                 "ns")

        # Verify that the bridge regenerated integrity correctly for each beat
        assert req["user"]["cmd_intg"] == get_cmd_intg(req,
                                                       width=device_if.width)
        assert req["user"]["data_intg"] == get_data_intg(req["data"],
                                                         width=device_if.width)

        received_reqs.append(req)

        # Create a response with correct integrity
        resp_beat = {
            "opcode": 0,
            "param": 0,
            "size": req["size"],
            "source": req["source"],
            "sink": 0,
            "data": 0,
            "error": 0
        }
        resp_beat["user"] = {
            "rsp_intg": get_rsp_intg(resp_beat, width=device_if.width),
            "data_intg": get_data_intg(0, width=device_if.width)
        }
        await device_if.device_d_fifo.put(resp_beat)

    # Receive the final assembled response on the host side
    resp = await with_timeout(host_if.host_get_response(), timeout_ns, "ns")

    # Verify that the bridge checked and regenerated integrity correctly
    expected_resp = resp.copy()
    expected_resp["error"] = 0
    assert resp["user"]["rsp_intg"] == get_rsp_intg(expected_resp,
                                                    width=host_if.width)
    assert resp["user"]["data_intg"] == get_data_intg(resp["data"],
                                                      width=host_if.width)
    assert resp["error"] == 0


@cocotb.test(timeout_time=20, timeout_unit="us")
async def test_16bit_writes_to_sram(dut):
    """Verify aligned writes to SRAM from both 32-bit and 128-bit hosts."""
    interfaces, clock = await setup_dut(dut)
    timeout_ns = TIMEOUT_CYCLES * clock.period
    device_if = interfaces["devices"][DEVICE_MAP["sram"]]

    mem = {}
    mismatches = 0

    async def device_responder():
        """Mock SRAM responder with memory storage."""
        device_width_bytes = device_if.width // 8
        while True:
            req = await device_if.device_get_request()
            addr = int(req["address"])
            aligned_base = addr & ~(device_width_bytes - 1)

            if int(req["opcode"]) in [0, 1]:  # PutFullData or PutPartialData
                data = int(req["data"])
                mask = int(req["mask"])
                for i in range(device_width_bytes):
                    if (mask >> i) & 1:
                        byte_val = (data >> (i * 8)) & 0xFF
                        mem[aligned_base + i] = byte_val
                await device_if.device_respond(
                    opcode=0,  # AccessAck
                    param=0,
                    size=req["size"],
                    source=req["source"],
                    width=device_if.width,
                )
            elif int(req["opcode"]) == 4:  # Get
                resp_data = 0
                for i in range(device_width_bytes):
                    byte_val = mem.get(aligned_base + i, 0)
                    resp_data |= byte_val << (i * 8)
                await device_if.device_respond(
                    opcode=1,  # AccessAckData
                    param=0,
                    size=req["size"],
                    source=req["source"],
                    data=resp_data,
                    width=device_if.width,
                )

    responder_task = cocotb.start_soon(device_responder())

    # --- 1. Test from 32-bit host (test_host_32) ---
    host_32_if = interfaces["hosts"][HOST_MAP["test_host_32"]]
    dut._log.info("--- Starting 32-bit host aligned write test ---")

    for i in range(4):
        addr = SRAM_BASE + i * 4
        val = 0xAAAA0000 + i

        # Aligned 32-bit write
        write_txn = create_a_channel_req(
            address=addr, data=val, mask=0xF, width=host_32_if.width, size=2
        )
        await host_32_if.host_put(write_txn)
        await host_32_if.host_get_response()

        # Aligned 32-bit read
        read_txn = create_a_channel_req(
            address=addr, width=host_32_if.width, is_read=True, size=2
        )
        await host_32_if.host_put(read_txn)
        resp = await host_32_if.host_get_response()

        read_val = int(resp["data"])
        if read_val != val:
            mismatches += 1
            dut._log.error(
                f"Mismatch at 32-bit host addr 0x{addr:X}: expected 0x{val:X}, got 0x{read_val:X}"
            )
        else:
            dut._log.info(f"Match at 32-bit host addr 0x{addr:X}: 0x{read_val:X}")

    # --- 2. Test from 128-bit host (coralnpu_core) ---
    host_128_if = interfaces["hosts"][HOST_MAP["coralnpu_core"]]
    dut._log.info("--- Starting 128-bit host 16-bit write test ---")

    for i in range(8):
        # We test sequential half-word writes.
        # i=0: addr=...100, mask=0x0003 (Byte 0,1)
        # i=1: addr=...102, mask=0x000C (Byte 2,3)
        # i=2: addr=...104, mask=0x0030 (Byte 4,5)
        # ...
        addr = SRAM_BASE + 0x100 + i * 2
        val = 0xB0B0 + i

        byte_offset = addr % 16
        mask = 0x3 << byte_offset
        data = val << (byte_offset * 8)

        dut._log.info(
            f"128-bit Host Write: addr=0x{addr:X} val=0x{val:X} (mask=0x{mask:X} data=0x{data:X})"
        )
        write_txn = create_a_channel_req(
            address=addr, data=data, mask=mask, width=host_128_if.width, size=1
        )
        await with_timeout(host_128_if.host_put(write_txn), timeout_ns, "ns")
        await with_timeout(host_128_if.host_get_response(), timeout_ns, "ns")

        # Read back and verify
        read_txn = create_a_channel_req(
            address=addr, width=host_128_if.width, is_read=True, size=1
        )
        await with_timeout(host_128_if.host_put(read_txn), timeout_ns, "ns")
        resp = await with_timeout(host_128_if.host_get_response(), timeout_ns, "ns")

        # We expect the 16-bit data to be at the byte_offset of the 128-bit word.
        read_val = (int(resp["data"]) >> (byte_offset * 8)) & 0xFFFF
        if read_val != val:
            mismatches += 1
            dut._log.error(
                f"Mismatch at 128-bit host addr 0x{addr:X}: expected 0x{val:X}, got 0x{read_val:X} (host_data=0x{int(resp['data']):X})"
            )
        else:
            dut._log.info(f"Match at 128-bit host addr 0x{addr:X}: 0x{read_val:X}")

    responder_task.cancel()
    assert mismatches == 0, f"Found {mismatches} mismatches during aligned write test."
    await ClockCycles(dut.clock, 10)


@cocotb.test(timeout_time=30, timeout_unit="us")
async def test_interleaved_hosts_to_sram(dut):
    """Reproduction: Verify that interleaved host transactions to a width bridge fail.

    This test triggers the source ID collision and aggregation bug in TlulWidthBridge
    by sending two 128-bit writes from different hosts (Core and SPI) and
    interleaving their acknowledgments on the D channel.
    """
    interfaces, clock = await setup_dut(dut)
    timeout_ns = TIMEOUT_CYCLES * clock.period
    device_if = interfaces["devices"][DEVICE_MAP["sram"]]

    # Host 0 (Core) and Host 1 (SPI)
    host0_if = interfaces["hosts"][HOST_MAP["coralnpu_core"]]
    host1_if = interfaces["hosts"][HOST_MAP["spi2tlul"]]

    async def interleaved_responder():
        """Mock SRAM responder that interleaves Acks for different host transactions."""
        # Wait for all 8 beats (4 for Host 0, 4 for Host 1)
        host0_reqs = []
        host1_reqs = []

        while len(host0_reqs) < 4 or len(host1_reqs) < 4:
            dut._log.info(
                f"Waiting for request... (H0: {len(host0_reqs)}, H1: {len(host1_reqs)})"
            )
            req = await device_if.device_get_request()
            dut._log.info(
                f"Received request: addr={hex(req['address'].integer)}, source={hex(req['source'].integer)}"
            )
            # Host 0 (Core) is assigned internal source IDs 0-3
            # Host 1 (SPI) is assigned internal source IDs 4-7
            if req["source"].integer < 4:
                host0_reqs.append(req)
            else:
                host1_reqs.append(req)

        dut._log.info("All requests received. Sending interleaved responses...")
        # Now interleave the responses on the D channel
        # Order: H0-B0, H1-B0, H0-B1, H1-B1, ...
        for i in range(4):
            # Ack Host 0 beat i
            dut._log.info(
                f"Responding to Host 0 beat {i}, source={hex(host0_reqs[i]['source'].integer)}"
            )
            await device_if.device_respond(
                opcode=0,
                param=0,
                size=host0_reqs[i]["size"],
                source=host0_reqs[i]["source"],
                width=device_if.width,
            )
            # Ack Host 1 beat i
            dut._log.info(
                f"Responding to Host 1 beat {i}, source={hex(host1_reqs[i]['source'].integer)}"
            )
            await device_if.device_respond(
                opcode=0,
                param=0,
                size=host1_reqs[i]["size"],
                source=host1_reqs[i]["source"],
                width=device_if.width,
            )

    responder_task = cocotb.start_soon(interleaved_responder())

    # Send 128-bit write from Host 0 to SRAM_BASE
    data0 = 0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    write0 = create_a_channel_req(
        address=SRAM_BASE, data=data0, mask=0xFFFF, width=host0_if.width, source=0
    )

    # Send 128-bit write from Host 1 to SRAM_BASE + 0x100
    data1 = 0x55555555555555555555555555555555
    write1 = create_a_channel_req(
        address=SRAM_BASE + 0x100,
        data=data1,
        mask=0xFFFF,
        width=host1_if.width,
        source=0,
    )  # Both use source ID 0

    # Put both requests into the pipeline
    await host0_if.host_put(write0)
    await host1_if.host_put(write1)

    # Wait for responses
    dut._log.info("Waiting for host responses...")

    async def get_resp0():
        res = await host0_if.host_get_response()
        dut._log.info(f"Host 0 got response: {res}")
        return res

    async def get_resp1():
        res = await host1_if.host_get_response()
        dut._log.info(f"Host 1 got response: {res}")
        return res

    resp0_task = cocotb.start_soon(get_resp0())
    resp1_task = cocotb.start_soon(get_resp1())

    try:
        resp0 = await with_timeout(resp0_task, timeout_ns, "ns")
        resp1 = await with_timeout(resp1_task, timeout_ns, "ns")
        dut._log.info(f"Host 0 Response: {resp0}")
        dut._log.info(f"Host 1 Response: {resp1}")
        assert resp0["error"] == 0
        assert resp1["error"] == 0
    except Exception as e:
        dut._log.error(f"Test failed with exception: {e}")
        raise e

    responder_task.cancel()


@cocotb.test(timeout_time=30, timeout_unit="us")
async def test_width_bridge_same_source_pipelined(dut):
    """Reproduction: Verify that back-to-back transactions with the same source ID work correctly.

    This test triggers the clear/write ordering bug in TlulWidthBridge by sending
    two 128-bit writes from the same host with the same source ID, and ensuring
    the first beat of the second transaction arrives from the device in the same
    cycle that the host accepts the response for the first transaction.
    """
    interfaces, clock = await setup_dut(dut)
    timeout_ns = TIMEOUT_CYCLES * clock.period
    device_if = interfaces["devices"][DEVICE_MAP["sram"]]
    host_if = interfaces["hosts"][HOST_MAP["coralnpu_core"]]

    # We use a source ID that is NOT used by other hosts to avoid interference.
    # coralnpu_core is host 0.
    test_source = 0

    data0 = 0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    write0 = create_a_channel_req(
        address=SRAM_BASE, data=data0, mask=0xFFFF, width=host_if.width, source=test_source
    )

    data1 = 0x55555555555555555555555555555555
    write1 = create_a_channel_req(
        address=SRAM_BASE + 0x10, data=data1, mask=0xFFFF, width=host_if.width, source=test_source
    )

    async def pipelined_responder():
        """Mock SRAM responder that provides beats back-to-back."""
        # Wait for all 8 beats (4 for write0, 4 for write1)
        reqs = []
        for _ in range(8):
            req = await device_if.device_get_request()
            reqs.append(req)

        # Send all 8 responses back-to-back.
        # The 5th response (first beat of write1) will be sent in the same cycle
        # as the bridge completes the 4th response (last beat of write0).
        for i in range(8):
            await device_if.device_respond(
                opcode=0,
                param=0,
                size=reqs[i]["size"],
                source=reqs[i]["source"],
                width=device_if.width,
            )

    responder_task = cocotb.start_soon(pipelined_responder())

    # Send both requests
    await host_if.host_put(write0)
    await host_if.host_put(write1)

    # Wait for first response
    resp0 = await with_timeout(host_if.host_get_response(), timeout_ns, "ns")
    assert resp0["error"] == 0
    assert resp0["source"] == test_source

    # Wait for second response
    # If the bug is present, the bridge might have cleared the state for test_source
    # in the same cycle it received the first beat of write1, causing write1
    # to never complete or have errors.
    resp1 = await with_timeout(host_if.host_get_response(), timeout_ns, "ns")
    assert resp1["error"] == 0
    assert resp1["source"] == test_source

    responder_task.cancel()
