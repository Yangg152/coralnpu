# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0

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

# ============ 模型数据 ============
generate_cc_arrays(
    name = "posenet_int8_cc",
    src = "models/posenet_075_353_481_int8.tflite",
    out = "posenet_int8.cc",
)

generate_cc_arrays(
    name = "posenet_int8_h",
    src = "models/posenet_075_353_481_int8.tflite",
    out = "posenet_int8.h",
)

cc_library(
    name = "posenet_model_lib",
    srcs = ["posenet_int8_cc"],
    hdrs = ["posenet_int8_h"],
    copts = [
        "-w",
        "-O0",
        "-Wno-error",
        "-fno-diagnostics-show-caret",
        "-fpermissive",
    ],
    visibility = ["//visibility:public"],
)

# ============ 输入图片数据 ============
generate_cc_arrays(
    name = "posenet_input_cc",
    src = "models/test_input_353x481.bmp",
    out = "posenet_input.cc",
)

generate_cc_arrays(
    name = "posenet_input_h",
    src = "models/test_input_353x481.bmp",
    out = "posenet_input.h",
)

cc_library(
    name = "posenet_input_lib",
    srcs = ["posenet_input_cc"],
    hdrs = ["posenet_input_h"],
    copts = [
        "-w",
        "-O0",
        "-Wno-error",
        "-fno-diagnostics-show-caret",
        "-fpermissive",
    ],
    visibility = ["//visibility:public"],
)

# ============ 主程序 ============
coralnpu_v2_binary(
    name = "posenet_int8_binary",
    srcs = [
        "run_posenet.cc",
    ],
    itcm_size_kbytes = 1024,
    dtcm_size_kbytes = 1024,
    deps = [
        ":posenet_model_lib",
        ":posenet_input_lib",
        "//sw/opt/litert-micro:conv",
        "//sw/opt/litert-micro:depthwise_conv",
        "//sw/opt/litert-micro:fully_connected",
        "//sw/opt/litert-micro:pooling",
        "//sw/opt/litert-micro:softmax",
        "//sw/opt/litert-micro:mxu",
        "@tflite_micro//tensorflow/lite/micro:micro_framework",
        "@tflite_micro//tensorflow/lite/micro:micro_log",
        "@tflite_micro//tensorflow/lite/micro:micro_profiler",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
        "@tflite_micro//tensorflow/lite/micro:system_setup",
    ],
)

cocotb_test_suite(
    name = "cocotb_run_posenet",
    simulators = [
        "verilator",
        "vcs",
    ],
    testcases = ["core_mini_rvv_posenet"],
    testcases_vname = "RUN_MOBILENET_FULL_TEST",
    tests_kwargs = {
        "waves": False,
        "tags": ["manual"],
        "hdl_toplevel": "RvvCoreMiniHighmemAxi",
        "seed": "42",
        "size": "enormous",
        "test_module": ["cocotb_posenet.py"],
        "deps": [
            "//coralnpu_test_utils:core_mini_axi_sim_interface",
            "//coralnpu_test_utils:sim_test_fixture",
            "@rules_python//python/runfiles",
        ],
        "data": ["posenet_int8_binary.elf"],
    },
    vcs_build_args = VCS_BUILD_ARGS,
    vcs_data = glob(["**/*.elf"]) + ["//tests/cocotb:coverage_exclude.cfg"],
    vcs_defines = VCS_DEFINES,
    vcs_test_args = VCS_TEST_ARGS,
    vcs_verilog_sources = ["//hdl/chisel/src/coralnpu:rvv_core_mini_highmem_axi_cc_library_verilog"],
    verilator_model = "//tests/cocotb:rvv_core_mini_highmem_axi_model",
)