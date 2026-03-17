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

generate_cc_arrays(
    name = "efficientnet_int8_cc",
    src = "models/efficientnet_int8.tflite",
    out = "efficientnet_int8.cc",
)

# 2. 生成 .h 文件
generate_cc_arrays(
    name = "efficientnet_int8_h",
    src = "models/efficientnet_int8.tflite",
    out = "efficientnet_int8.h",
)

# 3. 新增：将大数组独立成库，专门禁用警告(-w)和优化(-O0)
cc_library(
    name = "efficientnet_model_lib",
    srcs = ["efficientnet_int8_cc"],
    hdrs = ["efficientnet_int8_h"],
    copts = [
        "-w",                           # 1. 关闭所有警告
        "-O0",                          # 2. 关闭优化，防止编译器内存溢出
        "-Wno-error",                   # 3. 强制覆盖全局的 -Werror
        "-fno-diagnostics-show-caret",  # 🚨 4. 核心：禁止 GCC 打印超长的源代码行！
        "-fpermissive",                 # 5. 放宽语法检查
    ],
    visibility = ["//visibility:public"],
)

coralnpu_v2_binary(
    name = "efficientnet_int8_binary",
    srcs = [
        "run_efficientnet.cc", 
        # 注意：这里移除了 efficientnet_int8_cc
    ],
    # 注意：这里移除了 hdrs 属性
    itcm_size_kbytes = 1024,  
    dtcm_size_kbytes = 1024,  
    deps = [
        ":efficientnet_model_lib",  # 引入上面配置好的模型专用库
        "//sw/opt/litert-micro:mul",
        "//sw/opt/litert-micro:sub",
        "//sw/opt/litert-micro:conv",
        "//sw/opt/litert-micro:depthwise_conv",
        "//sw/opt/litert-micro:mean", 
        "//sw/opt/litert-micro:softmax",
        "@tflite_micro//tensorflow/lite/micro:micro_framework",
        "@tflite_micro//tensorflow/lite/micro:micro_log",
        "@tflite_micro//tensorflow/lite/micro:micro_profiler",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
        "@tflite_micro//tensorflow/lite/micro:system_setup",
    ],
)
cocotb_test_suite(
    name = "cocotb_run_efficientnet",
    simulators = [
        "verilator",
        "vcs",
    ],
    testcases = ["core_mini_rvv_efficientnet"],
    testcases_vname = "RUN_MOBILENET_FULL_TEST", 
    tests_kwargs = {
        "waves": False,
        "tags" : ["manual"],
        "hdl_toplevel": "RvvCoreMiniHighmemAxi",
        "seed": "42",
        "size": "enormous",
        "test_module": ["cocotb_efficientnet.py"], 
        "deps": [
            "//coralnpu_test_utils:core_mini_axi_sim_interface",
            "//coralnpu_test_utils:sim_test_fixture",
            "@rules_python//python/runfiles",
        ],
        "data": ["efficientnet_int8_binary.elf"], 
    },
    vcs_build_args = VCS_BUILD_ARGS,
    vcs_data = glob(["**/*.elf"]) + ["//tests/cocotb:coverage_exclude.cfg"],
    vcs_defines = VCS_DEFINES,
    vcs_test_args = VCS_TEST_ARGS,
    vcs_verilog_sources = ["//hdl/chisel/src/coralnpu:rvv_core_mini_highmem_axi_cc_library_verilog"],
    verilator_model = "//tests/cocotb:rvv_core_mini_highmem_axi_model",
)
