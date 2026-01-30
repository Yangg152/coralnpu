from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_v2_sim_utils import CoralNPUV2Simulator
import numpy as np

def run_mobilenet():
    npu_sim = CoralNPUV2Simulator(highmem_ld=True)
    r = runfiles.Create()
    elf_file = r.Rlocation('coralnpu_hw/tests/cocotb/tutorial/tfmicro/run_mobilenet_v1_025_partial_binary.elf')

    entry_point, symbol_map = npu_sim.get_elf_entry_and_symbol(elf_file, ['inference_status', 'inference_status_message'])
    npu_sim.load_program(elf_file, entry_point)
    npu_sim.run()
    npu_sim.wait()
    inference_status = npu_sim.read_memory(symbol_map['inference_status'], 1)[0]
    inference_status_message = npu_sim.read_memory(symbol_map['inference_status_message'], 31)
    inference_status_message = "".join([chr(i) for i in inference_status_message])
    print(f"inference_status {inference_status} and inference message is {inference_status_message}")

if __name__ == "__main__":
  run_mobilenet()

