#include "CPUAllocator.h"
#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <c10/core/alignment.h>
#include "alloc_cpu.h"


#include <c10/util/Logging.h>

// TODO: rename flag to C10
C10_DEFINE_bool(
    caffe2_report_cpu_memory_usage,
    false,
    "If set, print out detailed memory usage");

namespace c10 {
namespace distcxl {

struct C10_API DefaultCPUAllocator final : c10::Allocator {
  DefaultCPUAllocator() = default;
  DataPtr allocate(size_t nbytes) const override {
    void* data = nullptr;
    data = c10::distcxl::alloc_cpu(nbytes);
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::CPU)};
  }

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    c10::distcxl::free_cpu(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }
};


// QNNPACK AND XNNPACK may out-of-bound access the input and / or output
// tensors. This is by-design, and chosen to make the implementation of
// micro-kernels both simpler and faster as a result of not having to
// individually handle the corner cases where the number of processed elements
// is not a multiple of SIMD register width.  This behavior will trigger ASAN
// though, and may result in a segfault if the accessed memory location just so
// happens to fall on a page the current process has no read access to.  Here we
// define a custom allocator that allocates the extra storage required to keep
// this behavior safe.  This allocator could have been restricted to QNNPACK and
// XNNPACK only, but that would have negative performance ramifications, as
// input tensors must now be reallocated, and copied over, if the tensor is not
// allocated with this allocator to begin with.  Making this allocator the
// default on mobile builds minimizes the probability of unnecessary
// reallocations and copies, and also enables acceleration of operations where
// the output tensor is allocated outside of the function doing the
// implementation, wherein the implementation cannot simply re-allocate the
// output with the guarding allocator.
//
// PreGuardBytes: Number of guard bytes to allocate before the allocation.
// PostGuardBytes: Number of guard bytes to allocate after the allocation.

template <uint32_t PreGuardBytes, uint32_t PostGuardBytes>
class DefaultMobileCPUAllocator final : public c10::Allocator {
 public:
  DefaultMobileCPUAllocator() = default;
  ~DefaultMobileCPUAllocator() override = default;

  static void deleter(void* const pointer) {
    if (C10_UNLIKELY(!pointer)) {
      return;
    }
    // TODO: enable with better TLS support on mobile
    // profiledCPUMemoryReporter().Delete(pointer);
    c10::distcxl::free_cpu(pointer);
  }

  DataPtr allocate(const size_t nbytes) const override {
    if (C10_UNLIKELY(0u == nbytes)) {
      return {
          nullptr,
          nullptr,
          &deleter,
          at::Device(DeviceType::CPU),
      };
    }

    auto alloc_size = PreGuardBytes + nbytes + PostGuardBytes;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    void* data;
    data = c10::distcxl::alloc_cpu(alloc_size);
   
    return {
        reinterpret_cast<uint8_t*>(data) + PreGuardBytes,
        data,
        &deleter,
        at::Device(DeviceType::CPU),
    };
  }

  DeleterFnPtr raw_deleter() const override {
    return deleter;
  }
};

void NoDelete(void*) {}

c10::Allocator* GetCPUAllocator() {
  return GetAllocator(DeviceType::CPU);
}

void SetCPUAllocator(c10::Allocator* alloc, uint8_t priority) {
  SetAllocator(DeviceType::CPU, alloc, priority);
}

// The Mobile CPU allocator must always be present even on non-mobile builds
// because QNNPACK and XNNPACK are not mobile specific.
//
// Pre-guard: 8 bytes for QNNPACK, but set to gAlignment to ensure SIMD
//            alignment, not on the allocated memory, but memory location
//            returned to the user.
// Post-guard: 16 bytes for XNNPACK.

// NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-non-const-global-variables)
static DefaultMobileCPUAllocator<gAlignment, 16u> g_mobile_cpu_allocator;

c10::Allocator* GetDefaultMobileCPUAllocator() {
  return &g_mobile_cpu_allocator;
}

#ifdef C10_MOBILE

c10::Allocator* GetDefaultCPUAllocator() {
  return GetDefaultMobileCPUAllocator();
}

REGISTER_ALLOCATOR(DeviceType::CPU, &g_mobile_cpu_allocator);

#else

// Global default CPU Allocator
static DefaultCPUAllocator g_cpu_alloc;

c10::Allocator* GetDefaultCPUAllocator() {
  return &g_cpu_alloc;
}

REGISTER_ALLOCATOR(DeviceType::CPU, &g_cpu_alloc);

#endif /* C10_Mobile */

} // namespace distcxl
} // namespace c10
