#pragma once
#include <ATen/core/TensorBase.h>

namespace c10::distcxl {

inline void check_size_nonnegative(ArrayRef<int64_t> size) {
  for (const auto& x : size) {
    TORCH_CHECK(
        x >= 0,
        "Trying to create tensor with negative dimension ",
        x,
        ": ",
        size);
  }
}

inline void check_size_nonnegative(ArrayRef<c10::SymInt> size) {
  for (const auto& x : size) {
    TORCH_CHECK(
        x.expect_size(__FILE__, __LINE__),
        "Trying to create tensor with negative dimension ",
        x,
        ": ",
        size);
  }
}

TORCH_API size_t computeStorageNbytesContiguous(
    IntArrayRef sizes,
    size_t itemsize,
    size_t storage_offset = 0);
TORCH_API SymInt computeStorageNbytesContiguous(
    SymIntArrayRef sizes,
    const SymInt& itemsize,
    const SymInt& storage_offset = 0);
TORCH_API size_t computeStorageNbytes(
    IntArrayRef sizes,
    IntArrayRef strides,
    size_t itemsize,
    size_t storage_offset = 0);
TORCH_API SymInt computeStorageNbytes(
    SymIntArrayRef sizes,
    SymIntArrayRef strides,
    const SymInt& itemsize,
    const SymInt& storage_offset = 0);

TORCH_API at::TensorBase empty_generic(
    IntArrayRef size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    c10::optional<c10::MemoryFormat> memory_format_opt);

TORCH_API at::TensorBase empty_cpu_1(
    IntArrayRef size,
    ScalarType dtype,
    bool pin_memory = false,
    c10::optional<c10::MemoryFormat> memory_format_opt = c10::nullopt);

TORCH_API at::TensorBase empty_cpu_2(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt);

TORCH_API at::TensorBase empty_cpu_3(IntArrayRef size, const TensorOptions& options);

}
