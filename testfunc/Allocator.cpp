#include <c10/core/Allocator.h>
#include <c10/util/ThreadLocalDebugInfo.h>

namespace c10 {
namespace distcxl {

//static void deleteInefficientStdFunctionContext(void* ptr) {
//  delete static_cast<InefficientStdFunctionContext*>(ptr);
//}

//DataPtr InefficientStdFunctionContext::makeDataPtr(
//    void* ptr,
//    const std::function<void(void*)>& deleter,
//    Device device) {
//  return {
//      ptr,
//      new InefficientStdFunctionContext({ptr, deleter}),
//      &c10::distcxl::deleteInefficientStdFunctionContext,
//      device};
//}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
C10_API c10::Allocator* allocator_array[at::COMPILE_TIME_MAX_DEVICE_TYPES];
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
C10_API uint8_t allocator_priority[at::COMPILE_TIME_MAX_DEVICE_TYPES] = {0};

void SetAllocator(at::DeviceType t, c10::Allocator* alloc, uint8_t priority) {
  if (priority >= allocator_priority[static_cast<int>(t)]) {
    allocator_array[static_cast<int>(t)] = alloc;
    allocator_priority[static_cast<int>(t)] = priority;
  }
}

c10::Allocator* GetAllocator(const at::DeviceType& t) {
  auto* alloc = allocator_array[static_cast<int>(t)];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(alloc, "Allocator for ", t, " is not set.");
  return alloc;
}

bool memoryProfilingEnabled() {
  auto* reporter_ptr = static_cast<c10::MemoryReportingInfoBase*>(
      ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE));
  return reporter_ptr && reporter_ptr->memoryProfilingEnabled();
}

void reportMemoryUsageToProfiler(
    void* ptr,
    int64_t alloc_size,
    size_t total_allocated,
    size_t total_reserved,
    Device device) {
  auto* reporter_ptr = static_cast<c10::MemoryReportingInfoBase*>(
      ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE));
  if (reporter_ptr) {
    reporter_ptr->reportMemoryUsage(
        ptr, alloc_size, total_allocated, total_reserved, device);
  }
}

void reportOutOfMemoryToProfiler(
    int64_t alloc_size,
    size_t total_allocated,
    size_t total_reserved,
    Device device) {
  auto* reporter_ptr = static_cast<c10::MemoryReportingInfoBase*>(
      ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE));
  if (reporter_ptr) {
    reporter_ptr->reportOutOfMemory(
        alloc_size, total_allocated, total_reserved, device);
  }
}

} //namespace distcxl

c10::MemoryReportingInfoBase::MemoryReportingInfoBase() = default;

void c10::MemoryReportingInfoBase::reportOutOfMemory(
    int64_t /*alloc_size*/,
    size_t /*total_allocated*/,
    size_t /*total_reserved*/,
    Device /*device*/) {}

} // namespace c10
