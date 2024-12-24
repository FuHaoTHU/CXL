#pragma once

#include <c10/macros/Export.h>

#include <cstddef>

namespace c10 {
namespace distcxl {

C10_API void* alloc_cpu(size_t nbytes);
C10_API void free_cpu(void* data);

}
} // namespace c10
