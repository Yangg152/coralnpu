# file: rvv/Makefile

# ==========================================
# Tool
# ==========================================
VCS = vcs

# Directories
DESIGN_DIR = design
INC_DIR    = inc
COM_DIR    = common

# Include paths
INCDIRS = +incdir+$(INC_DIR) +incdir+$(DESIGN_DIR) +incdir+$(COM_DIR)

# Defines
DEFINES = +define+TB_SUPPORT

# ==========================================
# Source file list
# ==========================================
COM_SRCS = $(wildcard $(COM_DIR)/*.sv)

DESIGN_SRCS = \
	$(DESIGN_DIR)/Sram_256x128.v \
	$(DESIGN_DIR)/Aligner.sv \
	$(DESIGN_DIR)/MultiFifo.sv \
	$(DESIGN_DIR)/rvv_backend_vrf_reg.sv \
	$(DESIGN_DIR)/rvv_backend_vrf.sv \
	$(DESIGN_DIR)/rvv_backend_alu_unit_mask_viota.sv \
	$(DESIGN_DIR)/rvv_backend_alu_unit_addsub.sv \
	$(DESIGN_DIR)/rvv_backend_alu_unit_shift.sv \
	$(DESIGN_DIR)/rvv_backend_alu_unit_mask.sv \
	$(DESIGN_DIR)/rvv_backend_alu_unit_other.sv \
	$(DESIGN_DIR)/rvv_backend_alu_unit_execution_p1.sv \
	$(DESIGN_DIR)/rvv_backend_alu_unit.sv \
	$(DESIGN_DIR)/rvv_backend_alu.sv \
	$(DESIGN_DIR)/rvv_backend_mul_unit_mul8.sv \
	$(DESIGN_DIR)/rvv_backend_mul_unit.sv \
	$(DESIGN_DIR)/rvv_backend_mac_unit.sv \
	$(DESIGN_DIR)/rvv_backend_mulmac.sv \
	$(DESIGN_DIR)/rvv_backend_div_unit_divider.sv \
	$(DESIGN_DIR)/rvv_backend_div_unit.sv \
	$(DESIGN_DIR)/rvv_backend_div.sv \
	$(DESIGN_DIR)/rvv_backend_pmtrdt_unit.sv \
	$(DESIGN_DIR)/rvv_backend_pmtrdt.sv \
	$(DESIGN_DIR)/rvv_backend_mxu_pe.sv \
	$(DESIGN_DIR)/rvv_backend_mxu_unit.sv \
	$(DESIGN_DIR)/rvv_backend_mxu_wrapper.sv \
	$(DESIGN_DIR)/rvv_backend_lsu_remap.sv \
	$(DESIGN_DIR)/rvv_backend_decode_unit_ari.sv \
	$(DESIGN_DIR)/rvv_backend_decode_unit_lsu.sv \
	$(DESIGN_DIR)/rvv_backend_decode_unit.sv \
	$(DESIGN_DIR)/rvv_backend_decode_ctrl.sv \
	$(DESIGN_DIR)/rvv_backend_decode.sv \
	$(DESIGN_DIR)/rvv_backend_dispatch_raw_uop_rob.sv \
	$(DESIGN_DIR)/rvv_backend_dispatch_raw_uop_uop.sv \
	$(DESIGN_DIR)/rvv_backend_dispatch_structure_hazard.sv \
	$(DESIGN_DIR)/rvv_backend_dispatch_bypass.sv \
	$(DESIGN_DIR)/rvv_backend_dispatch_operand.sv \
	$(DESIGN_DIR)/rvv_backend_dispatch_opr_byte_type.sv \
	$(DESIGN_DIR)/rvv_backend_dispatch_ctrl.sv \
	$(DESIGN_DIR)/rvv_backend_dispatch.sv \
	$(DESIGN_DIR)/rvv_backend_retire.sv \
	$(DESIGN_DIR)/rvv_backend_rob.sv \
	$(DESIGN_DIR)/rvv_backend.sv

MXU_TOP_TB = mxu_top_tb.sv

# ==========================================
# Targets
# ==========================================
.PHONY: all run clean

all: compile run

compile:
	$(VCS) -full64 -sverilog -timescale=1ns/1ps \
		$(INCDIRS) $(DEFINES) \
		-top mxu_top_tb \
		-o simv \
		$(COM_SRCS) \
		$(DESIGN_SRCS) \
		$(MXU_TOP_TB) \
		-debug_access+all -kdb \
		+lint=TFIPC-L \
		-l compile.log 2>&1 | tee compile_screen.log

run: simv
	./simv -l run.log 2>&1 | tee run_screen.log

# Open waveform (if you dumped VCD)
wave:
	gtkwave mxu_top_tb.vcd &

clean:
	rm -rf simv simv.daidir csrc
	rm -rf *.log *.vcd ucli.key
	rm -rf DVEfiles inter.vpd vc_hdrs.h
	rm -rf .vlogansetup.args .vcs_lib_lock
