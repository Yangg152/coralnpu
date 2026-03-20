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

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "depthwise_conv",
    srcs = [
        "accumulator_util.h",
        "depthwise_conv.cc",
    ],
    hdrs = ["depthwise_conv.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"],
    deps = [
        ":memory_util",
        "//sw/opt:rvv_opt",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
    ],
)

cc_library(
    name = "memory_util",
    hdrs = ["memory_util.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"],
    deps = [
        "@tflite_micro//tensorflow/lite/kernels/internal:compatibility",
    ],
)

cc_library(
    name = "accumulator_util",
    hdrs = ["accumulator_util.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"],
    deps = [
        "@tflite_micro//tensorflow/lite/kernels/internal:compatibility",
    ],
)

cc_library(
    name = "mul",
    srcs = [
        "accumulator_util.h",
        "mul.cc",
    ],
    hdrs = ["mul.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"],
    deps = [
        "//sw/opt:rvv_opt",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
    ],
)

cc_library(
    name = "conv",
    srcs = [
        "accumulator_util.h",
        "conv.cc",
    ],
    hdrs = ["conv.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"], 
    deps = [
        ":memory_util",
        "//sw/opt:rvv_opt",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
    ],
)

cc_library(
    name = "sub",
    srcs = [
        "accumulator_util.h",
        "sub.cc",
    ],
    hdrs = ["sub.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"],
    deps = [
        "//sw/opt:rvv_opt",
        "@tflite_micro//tensorflow/lite/kernels/internal:reference",
        "@tflite_micro//tensorflow/lite/micro:micro_framework",
        "@tflite_micro//tensorflow/lite/micro/kernels:micro_ops",
    ],
)

cc_library(
    name = "mean",
    srcs = [
        "accumulator_util.h",
        "mean.cc",
    ],
    hdrs = ["mean.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"],
    deps = [
        "//sw/opt:rvv_opt",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
        "@tflite_micro//tensorflow/lite/kernels/internal:reference",
        "@tflite_micro//tensorflow/lite/micro:micro_framework",
        "@tflite_micro//tensorflow/lite/micro/kernels:micro_ops",
    ],
)

cc_library(
    name = "pooling",
    srcs = [
        "accumulator_util.h",
        "pooling.cc",
    ],
    hdrs = ["pooling.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"], 
    deps = [
        "//sw/opt:rvv_opt",
        "@tflite_micro//tensorflow/lite/kernels/internal:reference",
        "@tflite_micro//tensorflow/lite/micro:micro_framework",
        "@tflite_micro//tensorflow/lite/micro/kernels:micro_ops",
    ],
)

cc_library(
    name = "fully_connected",
    srcs = [
        "accumulator_util.h",
        "fully_connected.cc",
    ],
    hdrs = ["fully_connected.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"], 
    deps = [
        "//sw/opt:rvv_opt",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
        "@tflite_micro//tensorflow/lite/kernels/internal:reference",
        "@tflite_micro//tensorflow/lite/micro:micro_framework",
        "@tflite_micro//tensorflow/lite/micro/kernels:micro_ops",
    ],
)

cc_library(
    name = "softmax",
    srcs = [
        "accumulator_util.h",
        "softmax.cc",
    ],
    hdrs = ["softmax.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"], 
    deps = [
        "//sw/opt:rvv_opt",
        "@tflite_micro//tensorflow/lite/kernels/internal:reference",
        "@tflite_micro//tensorflow/lite/micro:micro_framework",
        "@tflite_micro//tensorflow/lite/micro/kernels:micro_ops",
    ],
)


cc_library(
    name = "mxu",
    srcs = [
        "accumulator_util.h",
        "mxu.cc",
    ],
    hdrs = [
        "mxu.h",
    ],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"], 
    deps = [
        ":memory_util",
        ":conv",         
        "//sw/opt:rvv_opt",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
    ],
)
