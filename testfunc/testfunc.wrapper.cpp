#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/torch.h>
#include "testfunc.h"
#include "CPUAllocator.h"
#include <c10/core/Allocator.h>
#include "alloc_cpu.h"
#include "numa.h"
#include <torch/extension.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Device.h>
#include <c10/core/TensorImpl.h>

namespace py = pybind11;


PYBIND11_MODULE(testfunc, m) {
    m.def("empty_cpu", [](py::tuple size_tuple,
                          c10::optional<at::ScalarType> dtype_opt = c10::nullopt,
                          c10::optional<at::Layout> layout_opt = c10::nullopt,
                          c10::optional<at::Device> device_opt = c10::nullopt,
                          c10::optional<bool> pin_memory_opt = c10::nullopt,
                          c10::optional<c10::MemoryFormat> memory_format_opt = c10::nullopt) {
        
        std::vector<int64_t> size_vector;
	for (auto& item : size_tuple) {
	    size_vector.push_back(py::cast<int64_t>(item));
	}
        c10::ArrayRef<int64_t> size(size_vector);

        if (!dtype_opt.has_value()) {
	   dtype_opt = at::kFloat;
	}
	if (!layout_opt.has_value()) {
	   layout_opt = at::kStrided;
	}
	if (!device_opt.has_value()) {
	   device_opt = at::Device(at::DeviceType::CPU);
	}
	if (!pin_memory_opt.has_value()) {
	   pin_memory_opt = true;
	}
	if (!memory_format_opt.has_value()) {
	   memory_format_opt = c10::MemoryFormat::Contiguous;
	}
        
        return c10::distcxl::empty_cpu_2(size, dtype_opt.value(), layout_opt.value(), device_opt.value(), pin_memory_opt.value(), memory_format_opt.value());
    }, "Create a tensor on CPU with the given size and options",
    py::arg("size"), 
    py::arg("dtype") = c10::nullopt, 
    py::arg("layout") = c10::nullopt, 
    py::arg("device") = c10::nullopt, 
    py::arg("pin_memory") = c10::nullopt, 
    py::arg("memory_format") = c10::nullopt 
    );
    m.def("alloc_cpu", &c10::distcxl::alloc_cpu, "Alloc on CPU");
    m.def("free_cpu", &c10::distcxl::free_cpu, "Free CPU Alloc");
    m.def("numa_check", &c10::distcxl::IsNUMAEnabled, "Check NUMA available");
    m.def("numa_find", &c10::distcxl::GetNUMANode, "Find which NUMA it is on");
    m.def("numa_move", &c10::distcxl::NUMAMove, "Move to New NUMA Node");
    m.def("numa_number", &c10::distcxl::GetNumNUMANodes, "Find NUMA counts");
}
