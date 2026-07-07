#ifndef FUSION_COMMON_ERROR_ERROR_CODE_H
#define FUSION_COMMON_ERROR_ERROR_CODE_H

#include <cstdint>
#include <string_view>

namespace fusion::error {
enum class ErrorDomain : std::uint8_t {
   Common,
   Alloc,
   Storage,
   Device,
   Planning,
   Fuir,
   Iter,
   Tensor,
   Topology,
   Ops,
   Autodiff,
   Python
};

enum class ErrorCategory : std::uint8_t {
   InvalidArgument,    // bad input/user request
   FailedPrecondition, // wrong state for requested operation
   Unsupported,        // valid request, but non-implemented/unsupported
   Internal,           // Internal invariant broken
   Unavailable         // backend/lib/resource unavailable
};

struct ErrorCode {
   ErrorDomain domain{};
   ErrorCategory category{};
   std::uint16_t detail{0};

   constexpr bool operator==(const ErrorCode &) const = default;
};

std::string_view to_string(ErrorDomain domain) noexcept;
std::string_view to_string(ErrorCategory category) noexcept;

} // namespace fusion::error

#endif // FUSION_COMMON_ERROR_ERROR_CODE_H