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

coralnpu_v2_binary(
    name = "run_mobilenet_v1_025_partial_binary",
    srcs = [
        "mobilenet_v1_025_partial_layers_cc",
        "run_mobilenet.cc",
    ],
    hdrs = ["mobilenet_v1_025_partial_layers.h"],
    itcm_size_kbytes = 1024,  # Highmem
    dtcm_size_kbytes = 1024,  # Highmem
    deps = [
        "//sw/opt/litert-micro:mul",
        "//sw/opt/litert-micro:sub",
        "//sw/opt/litert-micro:conv",
        "//sw/opt/litert-micro:depthwise_conv",
        "@tflite_micro//tensorflow/lite/micro:micro_framework",
        "@tflite_micro//tensorflow/lite/micro:micro_log",
        "@tflite_micro//tensorflow/lite/micro:micro_profiler",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
        "@tflite_micro//tensorflow/lite/micro:system_setup",
    ],
    visibility = ["//visibility:public"]
)

generate_cc_arrays(
    name = "mobilenet_v1_025_partial_layers_cc",
    src = "models/mobilenet_v1_025_partial_layers.tflite",
    out = "mobilenet_v1_025_partial_layers.cc",
)

generate_cc_arrays(
    name = "mobilenet_v1_025_partial_layers_h",
    src = "models/mobilenet_v1_025_partial_layers.tflite",
    out = "mobilenet_v1_025_partial_layers.h",
)

# BEGIN_TESTCASES_FOR_running_tflite
RUN_MOBILENET_TEST = [
    "core_mini_rvv_mobilenet_v1",
]
# END_TESTCASES_FOR_running_tflite

cocotb_test_suite(
    name = "cocotb_run_mobilenet_v1",
    simulators = [
        "verilator",
        "vcs",
    ],
    testcases = RUN_MOBILENET_TEST,
    testcases_vname = "RUN_MOBILENET_TEST",
    tests_kwargs = {
        "waves": False,
        "tags" : ["manual"],
        "hdl_toplevel": "RvvCoreMiniHighmemAxi",
        "seed": "42",
        "size": "enormous",
        "test_module": ["cocotb_run_mobilenet_v1.py"],
        "deps": [
            "//coralnpu_test_utils:core_mini_axi_sim_interface",
            "//coralnpu_test_utils:sim_test_fixture",
            "@rules_python//python/runfiles",
        ],
        "data": ["run_mobilenet_v1_025_partial_binary.elf"],
    },
    vcs_build_args = VCS_BUILD_ARGS,
    vcs_data = glob(["**/*.elf"]) + [
        "//tests/cocotb:coverage_exclude.cfg",
    ],
    vcs_defines = VCS_DEFINES,
    vcs_test_args = VCS_TEST_ARGS,
    vcs_verilog_sources = ["//hdl/chisel/src/coralnpu:rvv_core_mini_highmem_axi_cc_library_verilog"],
    verilator_model = "//tests/cocotb:rvv_core_mini_highmem_axi_model",
)

coralnpu_v2_binary(
    name = "depthwise_conv_test",
    srcs = ["depthwise_conv_test.cc"],
    itcm_size_kbytes = 1024,  # Highmem
    dtcm_size_kbytes = 1024,  # Highmem
    deps = [
        "//sw/opt/litert-micro:depthwise_conv",
        "@tflite_micro//tensorflow/lite/kernels/internal:reference_base"
    ],
)

cocotb_test_suite(
    name = "cocotb_depthwise_conv",
    simulators = [
        "verilator",
        "vcs",
    ],
    testcases = [
        "test_dwconv8to8stride1",
        "test_dwconv8to8stride2",
        ("test_dwconv32to32stride1", "large"),
        ("test_dwconv32to32stride2", "large"),
        ("test_dwconv64to64stride1", "large"),
        ("test_dwconv64to64stride2", "large"),
        ("test_dwconv16to32stride2", "large"),
        # Benchmarks are skipped.
    ],
    # testcases_vname = "RUN_MOBILENET_TEST",
    tests_kwargs = {
        "waves": False,
        "hdl_toplevel": "RvvCoreMiniHighmemAxi",
        "size": "enormous",
        "default_testcase_size": "medium",
        "test_module": ["cocotb_depthwise_conv.py"],
        "deps": [
            "//coralnpu_test_utils:core_mini_axi_sim_interface",
            "//coralnpu_test_utils:sim_test_fixture",
            "@rules_python//python/runfiles",
        ],
        "data": ["depthwise_conv_test.elf"],
    },
    vcs_build_args = VCS_BUILD_ARGS,
    vcs_data = glob(["**/*.elf"]) + [
        "//tests/cocotb:coverage_exclude.cfg",
    ],
    vcs_defines = VCS_DEFINES,
    vcs_test_args = VCS_TEST_ARGS,
    vcs_verilog_sources = ["//hdl/chisel/src/coralnpu:rvv_core_mini_highmem_axi_cc_library_verilog"],
    verilator_model = "//tests/cocotb:rvv_core_mini_highmem_axi_model",
)
# 1. 生成新模型的 C++ 数组文件
# 把 mobilenet_v1_025_128_quant.tflite 下载到了 models/ 目录下
generate_cc_arrays(
    name = "mobilenet_v1_025_128_quant_cc",
    src = "models/mobilenet_v1_025_128_quant.tflite",
    out = "mobilenet_v1_025_128_quant.cc",
)

generate_cc_arrays(
    name = "mobilenet_v1_025_128_quant_h",
    src = "models/mobilenet_v1_025_128_quant.tflite",
    out = "mobilenet_v1_025_128_quant.h",
)

# 2. 定义新的二进制文件目标
# 使用一个新的 C++ 源文件 run_mobilenet_full.cc
coralnpu_v2_binary(
    name = "run_mobilenet_v1_025_128_quant_binary",
    srcs = [
        "mobilenet_v1_025_128_quant_cc",
        "run_mobilenet_full.cc", 
    ],
    hdrs = ["mobilenet_v1_025_128_quant.h"],
    linker_script = "@coralnpu_hw//toolchain:coralnpu_tcm_highmem.ld",
    deps = [
        "//sw/opt/litert-micro:mul",
        "//sw/opt/litert-micro:sub",
        "//sw/opt/litert-micro:conv",
        "//sw/opt/litert-micro:depthwise_conv",
        "@tflite_micro//tensorflow/lite/micro:micro_framework",
        "@tflite_micro//tensorflow/lite/micro:micro_log",
        "@tflite_micro//tensorflow/lite/micro:micro_profiler",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
        "@tflite_micro//tensorflow/lite/micro:system_setup",
    ],
)

# 3. 更新测试套件
# 将新的 binary 添加到数据依赖中
cocotb_test_suite(
    name = "cocotb_run_mobilenet_full", # 改个名字区分
    simulators = [
        "verilator",
        "vcs",
    ],
    testcases = RUN_MOBILENET_TEST,
    testcases_vname = "RUN_MOBILENET_TEST",
    tests_kwargs = {
        "waves": False,
        "tags" : ["manual"],
        "hdl_toplevel": "RvvCoreMiniHighmemAxi",
        "seed": "42",
        "size": "enormous",
        "test_module": ["cocotb_run_mobilenet_v1.py"], # 复用 Python 脚本
        "deps": [
            "//coralnpu_test_utils:core_mini_axi_sim_interface",
            "//coralnpu_test_utils:sim_test_fixture",
            "@rules_python//python/runfiles",
        ],
        # 注意这里：改成新的 binary 名字
        "data": ["run_mobilenet_v1_025_128_quant_binary.elf"], 
    },
    vcs_build_args = VCS_BUILD_ARGS,
    vcs_data = glob(["**/*.elf"]) + ["//tests/cocotb:coverage_exclude.cfg"],
    vcs_defines = VCS_DEFINES,
    vcs_test_args = VCS_TEST_ARGS,
    vcs_verilog_sources = ["//hdl/chisel/src/coralnpu:rvv_core_mini_highmem_axi_cc_library_verilog"],
    verilator_model = "//tests/cocotb:rvv_core_mini_highmem_axi_model",
)

# 1. 编译 C++ 测试二进制
coralnpu_v2_binary(
    name = "conv_test", # 生成 conv_test.elf
    srcs = ["conv_test.cc"],
    linker_script = "@coralnpu_hw//toolchain:coralnpu_tcm_highmem.ld",
    deps = [
        "//sw/opt/litert-micro:conv", 
        "//sw/opt/litert-micro:depthwise_conv", # 借用一下配置
        "@tflite_micro//tensorflow/lite/kernels/internal:reference_base"
    ],
)

# 2. 定义 Python 测试套件
cocotb_test_suite(
    name = "cocotb_conv",
    simulators = [
        "verilator",
        "vcs",
    ],
    testcases = [
        "test_conv1x1_32to32",
        "test_conv1x1_32to64",
        "test_conv1x1_64to32",
        "test_conv1x1_odd_channels",
    ],
    tests_kwargs = {
        "waves": False, # 调试时可以设为 True
        "hdl_toplevel": "RvvCoreMiniHighmemAxi",
        "size": "enormous",
        "default_testcase_size": "medium",
        "test_module": ["cocotb_conv.py"],
        "deps": [
            "//coralnpu_test_utils:core_mini_axi_sim_interface",
            "//coralnpu_test_utils:sim_test_fixture",
            "@rules_python//python/runfiles",
        ],
        "data": ["conv_test.elf"],
    },
    vcs_build_args = VCS_BUILD_ARGS,
    vcs_data = glob(["**/*.elf"]) + [
        "//tests/cocotb:coverage_exclude.cfg",
    ],
    vcs_defines = VCS_DEFINES,
    vcs_test_args = VCS_TEST_ARGS,
    vcs_verilog_sources = ["//hdl/chisel/src/coralnpu:rvv_core_mini_highmem_axi_cc_library_verilog"],
    verilator_model = "//tests/cocotb:rvv_core_mini_highmem_axi_model",
)

# mul and sub
coralnpu_v2_binary(
    name = "mul_sub_test", 
    srcs = ["mul_sub_test.cc"],
    linker_script = "@coralnpu_hw//toolchain:coralnpu_tcm_highmem.ld",
    deps = [
        "//sw/opt/litert-micro:mul", 
        "//sw/opt/litert-micro:sub", 
        "//sw/opt/litert-micro:conv", 
        "//sw/opt/litert-micro:depthwise_conv", 
        "@tflite_micro//tensorflow/lite/micro:micro_framework",
        "@tflite_micro//tensorflow/lite/micro:micro_log",
        "@tflite_micro//tensorflow/lite/micro:micro_profiler",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
        "@tflite_micro//tensorflow/lite/micro:system_setup",
    ],
)

# 2. 定义 Python 测试套件
cocotb_test_suite(
    name = "cocotb_mul_sub",
    simulators = [
        "verilator",
        "vcs",
    ],
    testcases = [
        "test_mul_elementwise_small",
        "test_sub_elementwise_small",
        "test_mul_elementwise",
        "test_mul_broadcast",
        "test_sub_elementwise",
        "test_sub_broadcast_vector_minus_scalar",
        "test_sub_broadcast_scalar_minus_vector",
    ],
    tests_kwargs = {
        "waves": False, # 调试时可以设为 True
        "hdl_toplevel": "RvvCoreMiniHighmemAxi",
        "size": "enormous",
        "default_testcase_size": "medium",
        "test_module": ["cocotb_mul_sub.py"],
        "deps": [
            "//coralnpu_test_utils:core_mini_axi_sim_interface",
            "//coralnpu_test_utils:sim_test_fixture",
            "@rules_python//python/runfiles",
        ],
        "data": ["mul_sub_test.elf"],
    },
    vcs_build_args = VCS_BUILD_ARGS,
    vcs_data = glob(["**/*.elf"]) + [
        "//tests/cocotb:coverage_exclude.cfg",
    ],
    vcs_defines = VCS_DEFINES,
    vcs_test_args = VCS_TEST_ARGS,
    vcs_verilog_sources = ["//hdl/chisel/src/coralnpu:rvv_core_mini_highmem_axi_cc_library_verilog"],
    verilator_model = "//tests/cocotb:rvv_core_mini_highmem_axi_model",
)