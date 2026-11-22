from hardware_model.hw_modules import IOModule, IO_module_dict, MemoryModule, memory_module_dict, ComputeModule, compute_module_dict

class Device:
    def __init__(
        self,
        compute_module: ComputeModule,
        io_module: IOModule,
        memory_module: MemoryModule,
    ) -> None:
        self.compute_module = compute_module
        self.io_module = io_module
        self.memory_module = memory_module


device_dict = {
    "A100_80GB_fp16": Device(
        compute_module_dict["A100_fp16"],
        IO_module_dict["A100"],
        memory_module_dict["A100_80GB"],
    ),
    "NPU_int8": Device(
        compute_module_dict["NPU_int8"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "NPU_fp16": Device(
        compute_module_dict["NPU_fp16"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "NPU_fp32": Device(
        compute_module_dict["NPU_fp32"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "MI210": Device(
        compute_module_dict["MI210_fp16"],
        IO_module_dict["MI210"],
        memory_module_dict["MI210"],
    ),
    "PIM_int8": Device(
        compute_module_dict["PIM_int8"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "PIM_fp16": Device(
        compute_module_dict["PIM_fp16"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "PIM_fp32": Device(
        compute_module_dict["PIM_fp32"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "SIGMA_int8": Device(
        compute_module_dict["SIGMA_int8"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "SIGMA_fp16": Device(
        compute_module_dict["SIGMA_fp16"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
    "SIGMA_fp32": Device(
        compute_module_dict["SIGMA_fp32"],
        IO_module_dict["NPU"],
        memory_module_dict["NPU"],
    ),
}
