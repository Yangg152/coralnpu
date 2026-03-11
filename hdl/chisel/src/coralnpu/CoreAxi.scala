// Copyright 2024 Google LLC
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

package coralnpu

import chisel3._
import chisel3.util._

import bus._
import common._

class CoreAxi(p: Parameters, coreModuleName: String) extends RawModule {
  override val desiredName = coreModuleName + "Axi"
  val memoryRegions = p.m
  val io = IO(new Bundle {
    // AXI
    val aclk = Input(Clock())
    val aresetn = Input(AsyncReset())
    // ITCM, DTCM, CSR
    val axi_slave = Flipped(new AxiMasterIO(p.axi2AddrBits, p.axi2DataBits, p.axi2IdBits))
    val axi_master = new AxiMasterIO(p.axi2AddrBits, p.axi2DataBits, p.axi2IdBits)
    // Core status interrupts
    val halted = Output(Bool())
    val fault = Output(Bool())
    val wfi = Output(Bool())
    val irq = Input(Bool())
    // Debug data interface
    val debug = new DebugIO(p)
    val dm = new DebugModuleIO(p)
    val te = Input(Bool())
  })
  dontTouch(io)

  val rst_sync = Module(new RstSync())
  rst_sync.io.clk_i := io.aclk
  rst_sync.io.rstn_i := io.aresetn
  rst_sync.io.clk_en := true.B
  rst_sync.io.te := io.te

  val global_reset = (!Mux(io.te, io.aresetn, rst_sync.io.rstn_o).asBool).asAsyncReset
  withClockAndReset(rst_sync.io.clk_o, global_reset) {
    // Build CSR
    val csr = Module(new CoreCSR(p))
    csr.io.internal := false.B

    // Build core and connect with CSR
    val cg = Module(new ClockGate)
    cg.io.clk_i := rst_sync.io.clk_o
    cg.io.te := io.te

    val dm = Module(new DebugModule(p))
        dontTouch(dm.io)
    val dmEnable = RegInit(false.B)
    dmEnable := true.B
    val dmReqArbiter = Module(new CoralNPURRArbiter(new DebugModuleReqIO(p), 2))
    dmReqArbiter.io.in(0) <> GateDecoupled(io.dm.req, dmEnable)
    dmReqArbiter.io.in(1) <> GateDecoupled(csr.io.debug.req, dmEnable)

    // Queue source ID of the request to route the response later.
    val inflight = Module(new Queue(UInt(1.W), 1))

    dm.io.ext.req.bits := dmReqArbiter.io.out.bits
    dm.io.ext.req.valid := dmReqArbiter.io.out.valid && inflight.io.enq.ready
    dmReqArbiter.io.out.ready := dm.io.ext.req.ready && inflight.io.enq.ready

    inflight.io.enq.bits := dmReqArbiter.io.chosen
    inflight.io.enq.valid := dmReqArbiter.io.out.valid && dm.io.ext.req.ready

    val rspId = inflight.io.deq.bits
    inflight.io.deq.ready := dm.io.ext.rsp.fire

    csr.io.debug.rsp.bits := dm.io.ext.rsp.bits
    io.dm.rsp.bits := dm.io.ext.rsp.bits

    csr.io.debug.rsp.valid := dm.io.ext.rsp.valid && inflight.io.deq.valid && (rspId === 1.U)
    io.dm.rsp.valid := dm.io.ext.rsp.valid && inflight.io.deq.valid && (rspId === 0.U)

    dm.io.ext.rsp.ready := inflight.io.deq.valid && Mux(rspId === 1.U, csr.io.debug.rsp.ready, io.dm.rsp.ready)
    
    val core_reset = Mux(io.te, (!io.aresetn.asBool).asAsyncReset, (csr.io.reset || dm.io.ndmreset).asAsyncReset)
    val core = withClockAndReset(cg.io.clk_o, core_reset) { Core(p, coreModuleName) }
    cg.io.enable := io.irq || (!csr.io.cg && !core.io.wfi) || dm.io.haltreq(0)
    io.halted := core.io.halted
    io.fault := core.io.fault
    io.wfi := core.io.wfi
    core.io.irq := io.irq || dm.io.haltreq(0)
    csr.io.halted := core.io.halted
    csr.io.fault := core.io.fault
    csr.io.coralnpu_csr := core.io.csr.out
    core.io.debug_req := true.B
    core.io.csr.in.value(0) := csr.io.pcStart
    for (i <- 1 until p.csrInCount) {
      core.io.csr.in.value(i) := 0.U
    }
    io.debug <> core.io.debug
    // Tie-offs (no cache to flush)
    core.io.dflush.ready := true.B
    core.io.iflush.ready := true.B


    core.io.dm.debug_req := dm.io.haltreq(0)
    core.io.dm.resume_req := dm.io.resumereq(0)
    dm.io.resumeack(0) := !core.io.dm.debug_mode && RegNext(core.io.dm.debug_mode, false.B)
    dm.io.halted(0) := core.io.dm.debug_mode
    dm.io.running(0) := !core.io.dm.debug_mode
    dm.io.havereset(0) := false.B
    core.io.dm.csr := dm.io.csr
    core.io.dm.csr_rs1 := dm.io.csr_rs1
    dm.io.csr_rd := core.io.dm.csr_rd
    dm.io.scalar_rd <> core.io.dm.scalar_rd
    dm.io.scalar_rs <> core.io.dm.scalar_rs
    if (p.enableFloat) {
      dm.io.float_rd.get <> core.io.dm.float_rd.get
      dm.io.float_rs.get <> core.io.dm.float_rs.get
    }

    // TCMs are arbitrated between the core ({i|d}bus), the AXI slave interface,
    // and the debug module.
    val tcmPortCount = 3

    // Build ITCM and connect to ibus
    val itcmSizeBytes: Int = 1024 * p.itcmSizeKBytes
    val itcmSubEntryWidth = 8
    val itcmWidth = p.axi2DataBits
    val itcmEntries = itcmSizeBytes / (itcmWidth / 8)
    val itcm = Module(new TCM128(itcmSizeBytes, itcmSubEntryWidth))
    dontTouch(itcm.io)
    val itcmWrapper = Module(new SRAM(p, log2Ceil(itcmEntries)))
    itcm.io.addr := itcmWrapper.io.sram.address
    itcm.io.enable := itcmWrapper.io.sram.enable
    itcm.io.write := itcmWrapper.io.sram.isWrite
    itcm.io.wdata := itcmWrapper.io.sram.writeData
    itcm.io.wmask := itcmWrapper.io.sram.mask
    itcmWrapper.io.sram.readData := itcm.io.rdata
    val itcmArbiter = Module(new FabricArbiter(p, tcmPortCount))
    itcmArbiter.io.port <> itcmWrapper.io.fabric
    itcmArbiter.io.source(0).readDataAddr := MakeValid(
        core.io.ibus.valid, core.io.ibus.addr)
    itcmArbiter.io.source(0).writeDataAddr :=
        MakeInvalid(UInt(p.axi2AddrBits.W))
    itcmArbiter.io.source(0).writeDataBits := 0.U
    itcmArbiter.io.source(0).writeDataStrb := 0.U
    core.io.ibus.rdata := itcmArbiter.io.source(0).readData.bits
    core.io.ibus.ready := true.B  // Can always read from TCM
    /// Connect fault for the ibus.
    core.io.ibus.fault.valid :=
        core.io.ibus.valid && !(memoryRegions(0).contains(core.io.ibus.addr))
    core.io.ibus.fault.bits.write := false.B
    core.io.ibus.fault.bits.addr := 0.U
    core.io.ibus.fault.bits.epc := core.io.ibus.addr

    // Build DTCM and connect to dbus
    val dtcmSizeBytes: Int = 1024 * p.dtcmSizeKBytes
    val dtcmWidth = p.axi2DataBits
    val dtcmEntries = dtcmSizeBytes / (dtcmWidth / 8)
    val dtcmSubEntryWidth = 8
    val dtcm = Module(new TCM128(dtcmSizeBytes, dtcmSubEntryWidth))
    dontTouch(dtcm.io)
    val dtcmWrapper = Module(new SRAM(p, log2Ceil(dtcmEntries)))
    dtcm.io.addr := dtcmWrapper.io.sram.address
    dtcm.io.enable := dtcmWrapper.io.sram.enable
    dtcm.io.write := dtcmWrapper.io.sram.isWrite
    dtcm.io.wdata := dtcmWrapper.io.sram.writeData
    dtcm.io.wmask := dtcmWrapper.io.sram.mask
    dtcmWrapper.io.sram.readData := dtcm.io.rdata
    val dtcmArbiter = Module(new FabricArbiter(p, tcmPortCount))
    dtcmArbiter.io.port <> dtcmWrapper.io.fabric
    dtcmArbiter.io.source(0).readDataAddr := MakeValid(
        core.io.dbus.valid && !core.io.dbus.write, core.io.dbus.addr)
    dtcmArbiter.io.source(0).writeDataAddr := MakeValid(
        core.io.dbus.valid && core.io.dbus.write, core.io.dbus.addr)
    dtcmArbiter.io.source(0).writeDataBits := core.io.dbus.wdata
    dtcmArbiter.io.source(0).writeDataStrb := core.io.dbus.wmask
    core.io.dbus.rdata := dtcmArbiter.io.source(0).readData.bits
    core.io.dbus.ready := true.B  // Can always read/write TCM

    // Connect TCMs and CSR into fabric
    val fabricMux = Module(new FabricMux(p, memoryRegions))
    fabricMux.io.ports(0) <> itcmArbiter.io.source(1)
    fabricMux.io.periBusy(0) := itcmArbiter.io.fabricBusy(1)
    fabricMux.io.ports(1) <> dtcmArbiter.io.source(1)
    fabricMux.io.periBusy(1) := dtcmArbiter.io.fabricBusy(1)
    fabricMux.io.ports(2) <> csr.io.fabric
    fabricMux.io.periBusy(2) := false.B

    
      itcmArbiter.io.source(2) <> dm.io.itcm
      dtcmArbiter.io.source(2) <> dm.io.dtcm

    // Create AXI Slave interface and connect internal fabric to AXI
    val axiSlave = Module(new AxiSlave(p))
    val axiSlaveEnable = RegInit(false.B)
    axiSlaveEnable := true.B
    axiSlave.io.fabric <> fabricMux.io.source
    axiSlave.io.periBusy := fabricMux.io.fabricBusy
    axiSlave.io.axi.write.addr <> GateDecoupled(io.axi_slave.write.addr, axiSlaveEnable)
    axiSlave.io.axi.write.data <> GateDecoupled(io.axi_slave.write.data, axiSlaveEnable)
    io.axi_slave.write.resp <> GateDecoupled(axiSlave.io.axi.write.resp, axiSlaveEnable)
    axiSlave.io.axi.read.addr <> GateDecoupled(io.axi_slave.read.addr, axiSlaveEnable)
    io.axi_slave.read.data <> GateDecoupled(axiSlave.io.axi.read.data, axiSlaveEnable)

    // Connect ebus to AXI Master
    val ebus2axi = DBus2Axi(p)
    ebus2axi.io.dbus <> core.io.ebus.dbus
    ebus2axi.io.axi <> io.axi_master
    ebus2axi.io.fault <> core.io.ebus.fault
  }
}
