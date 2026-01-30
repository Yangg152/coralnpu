// Copyright 2026 Google LLC
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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "sim/coralnpu_v2_simulator.h"

using coralnpu::sim::CoralNPUV2LsuAccessRange;
using coralnpu::sim::CoralNPUV2Simulator;
using coralnpu::sim::CoralNPUV2SimulatorOptions;

namespace py = pybind11;

class CoralNPUV2SimulatorPy {
 public:
  explicit CoralNPUV2SimulatorPy(
      const coralnpu::sim::CoralNPUV2SimulatorOptions& options);
  ~CoralNPUV2SimulatorPy() = default;

  void LoadProgram(const std::string& elf_file_path,
                   std::optional<uint32_t> entry_point = std::nullopt);
  void Run();
  void Wait();
  int Step(const int num_steps);
  uint64_t GetCycleCount();
  uint64_t ReadRegister(const std::string& name);
  py::array_t<uint8_t> ReadMemory(uint64_t address, size_t length);
  void WriteMemory(uint64_t address, py::array_t<uint8_t> input_buffer,
                   size_t length);

 private:
  coralnpu::sim::CoralNPUV2Simulator sim_;
};

void CoralNPUV2SimulatorPy::LoadProgram(const std::string& elf_file_path,
                                        std::optional<uint32_t> entry_point) {
  LOG(INFO) << "Calling LoadProgram...";
  absl::Status status = sim_.LoadProgram(elf_file_path, entry_point);
  if (!status.ok()) {
    LOG(ERROR) << "LoadProgram failed: " << status;
    return;
  }
}

void CoralNPUV2SimulatorPy::Run() {
  LOG(INFO) << "Calling Run...";
  absl::Status status = sim_.Run();
  if (!status.ok()) {
    LOG(ERROR) << "Run failed: " << status;
    return;
  }
}

void CoralNPUV2SimulatorPy::Wait() {
  LOG(INFO) << "Calling Wait...";
  absl::Status status = sim_.Wait();
  if (!status.ok()) {
    LOG(ERROR) << "Wait failed: " << status;
    return;
  }
}

int CoralNPUV2SimulatorPy::Step(const int num_steps) {
  absl::StatusOr<int> count_status = sim_.Step(num_steps);
  if (!count_status.ok()) {
    LOG(ERROR) << "Step failed: " << count_status;
    return -1;
  }
  return count_status.value();
}

uint64_t CoralNPUV2SimulatorPy::GetCycleCount() {
  uint64_t cycle_count = sim_.GetCycleCount();
  if (cycle_count) {
    return cycle_count;
  }
  return 0;
}

uint64_t CoralNPUV2SimulatorPy::ReadRegister(const std::string& name) {
  absl::StatusOr<uint64_t> word_status = sim_.ReadRegister(name);
  if (!word_status.ok()) {
    LOG(ERROR) << "ReadRegister failed: " << word_status;
  }
  return word_status.value();
}

void CoralNPUV2SimulatorPy::WriteMemory(uint64_t address,
                                        py::array_t<uint8_t> input_buffer,
                                        size_t length) {
  py::buffer_info info = input_buffer.request();
  const void* input_buffer_ptr = static_cast<const void*>(info.ptr);

  absl::StatusOr<size_t> write_status =
      sim_.WriteMemory(address, input_buffer_ptr, length);
  if (!write_status.ok()) {
    LOG(ERROR) << "Write memory failed: " << write_status;
  }
  if (write_status.value() != length) {
    LOG(ERROR) << "Write memory length assertion error: "
               << write_status.value() << " written out of" << length;
  }
}

py::array_t<uint8_t> CoralNPUV2SimulatorPy::ReadMemory(uint64_t address,
                                                       size_t length) {
  auto result = py::array_t<uint8_t>(length);
  py::buffer_info info = result.request();
  void* buffer_ptr = info.ptr;

  absl::StatusOr<size_t> read_status =
      sim_.ReadMemory(address, buffer_ptr, length);
  if (!read_status.ok()) {
    LOG(ERROR) << "Read memory failed: " << read_status.status();
  } else if (read_status.value() != length) {
    LOG(ERROR) << "Read memory length assertion error: " << read_status.value()
               << " read out of " << length;
  }
  return result;
}

CoralNPUV2SimulatorPy::CoralNPUV2SimulatorPy(
    const coralnpu::sim::CoralNPUV2SimulatorOptions& options)
    : sim_(options) {}

PYBIND11_MODULE(coralnpu_v2_sim_pybind, module) {
  module.doc() =
      "This module is created with pybind wrap on coralnpu-v2-simulator";
  py::class_<CoralNPUV2LsuAccessRange>(module, "CoralNPUV2LsuAccessRange")
      .def(py::init<>())
      .def_readwrite("start_address", &CoralNPUV2LsuAccessRange::start_address)
      .def_readwrite("length", &CoralNPUV2LsuAccessRange::length);

  py::class_<CoralNPUV2SimulatorOptions>(module, "CoralNPUV2SimulatorOptions")
      .def(py::init<>())
      .def_readwrite("itcm_start_address",
                     &CoralNPUV2SimulatorOptions::itcm_start_address)
      .def_readwrite("itcm_length", &CoralNPUV2SimulatorOptions::itcm_length)
      .def_readwrite("initial_misa_value",
                     &CoralNPUV2SimulatorOptions::initial_misa_value)
      .def_readwrite("lsu_access_ranges",
                     &CoralNPUV2SimulatorOptions::lsu_access_ranges)
      .def_readwrite("exit_on_ebreak",
                     &CoralNPUV2SimulatorOptions::exit_on_ebreak);

  py::class_<CoralNPUV2SimulatorPy>(module, "CoralNPUV2SimulatorPy")
      .def(py::init<const CoralNPUV2SimulatorOptions>())
      .def("LoadProgram", &CoralNPUV2SimulatorPy::LoadProgram,
           py::arg("elf_file_path"), py::arg("entry_point"), "loads elf")
      .def("Run", &CoralNPUV2SimulatorPy::Run, "runs the program")
      .def("Wait", &CoralNPUV2SimulatorPy::Wait,
           "waits for the program to finish")
      .def("Step", &CoralNPUV2SimulatorPy::Step, py::arg("num_steps"),
           "Runs a simulator to a specifed number of steps")
      .def("GetCycleCount", &CoralNPUV2SimulatorPy::GetCycleCount,
           "Return number of cycles taken by program to run")
      .def("ReadMemory", &CoralNPUV2SimulatorPy::ReadMemory,
           "Reads memory and stores in the given buffer")
      .def("WriteMemory", &CoralNPUV2SimulatorPy::WriteMemory, "Write memory")
      .def("ReadRegister", &CoralNPUV2SimulatorPy::ReadRegister,
           "Read Register and returns the value in it");
}