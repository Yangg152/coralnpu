# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

load("@rules_hdl//verilog:providers.bzl", "verilog_library")

package(default_visibility = ["//visibility:public"])

verilog_library(
    name = "tl_test_pkg",
    srcs = ["tl_test_pkg.sv"],
)

filegroup(
    name = "core_files",
    srcs = glob([
        "*.core",
        "*.sv",
    ]),
)
