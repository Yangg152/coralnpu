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

load("@coralnpu_hw//third_party/python:requirements.bzl", "requirement")
load("//rules:coco_tb.bzl", "cocotb_test_suite", "verilator_cocotb_model")
load("//rules:coralnpu_v2.bzl", "coralnpu_v2_binary")
load("//rules:utils.bzl", "generate_cc_arrays")
load(
    "//tests/cocotb:build_defs.bzl",
    "VCS_BUILD_ARGS",
    "VCS_DEFINES",
    "VCS_TEST_ARGS",
    "VERILATOR_BUILD_ARGS",
)

# ==============================================================================
# VWW (Visual Wake Words)
# ==============================================================================
coralnpu_v2_binary(
    name = "run_vww_binary",
    srcs = [
        "vww/run_vww.cpp",
        "vww/vww_data/vww_model_data.cc",
        "vww/vww_data/vww_input_data.cc",
        "vww/vww_data/vww_model_settings.cc",
    ],
    hdrs = [
        "vww/vww_data/vww_model_data.h",
        "vww/vww_data/vww_input_data.h",
        "vww/vww_data/vww_model_settings.h",
    ],
    itcm_size_kbytes = 1024,
    dtcm_size_kbytes = 1024,
    deps = [
        "//sw/opt/litert-micro:conv",
        "//sw/opt/litert-micro:depthwise_conv",
        "//sw/opt/litert-micro:fully_connected",
        "//sw/opt/litert-micro:pooling",
        "//sw/opt/litert-micro:softmax",
        # "//third_party/muriscv-nn:muriscv_nn_tflm_kernels",
        "@tflite_micro//tensorflow/lite/micro:micro_framework",
        "@tflite_micro//tensorflow/lite/micro:micro_log",
        "@tflite_micro//tensorflow/lite/micro:micro_profiler",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
        "@tflite_micro//tensorflow/lite/micro:system_setup",
    ],
)

cocotb_test_suite(
    name = "cocotb_run_vww",
    simulators = [
        "verilator",
        "vcs",
    ],
    testcases = ["core_mini_rvv_vww_full"],
    testcases_vname = "RUN_VWW_TEST",
    tests_kwargs = {
        "waves": False,
        "tags": ["manual"],
        "hdl_toplevel": "RvvCoreMiniHighmemAxi",
        "seed": "42",
        "size": "enormous",
        "test_module": ["cocotb_run_vww.py"],
        "deps": [
            "//coralnpu_test_utils:core_mini_axi_sim_interface",
            "//coralnpu_test_utils:sim_test_fixture",
            "@rules_python//python/runfiles",
        ],
        "data": ["run_vww_binary.elf"],
    },
    vcs_build_args = VCS_BUILD_ARGS,
    vcs_data = glob(["**/*.elf"]) + ["//tests/cocotb:coverage_exclude.cfg"],
    vcs_defines = VCS_DEFINES,
    vcs_test_args = VCS_TEST_ARGS,
    vcs_verilog_sources = ["//hdl/chisel/src/coralnpu:rvv_core_mini_highmem_axi_cc_library_verilog"],
    verilator_model = "//tests/cocotb:rvv_core_mini_highmem_axi_model",
)