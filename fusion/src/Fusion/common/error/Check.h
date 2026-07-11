#ifndef FUSION_COMMON_ERROR_CHECK_H
#define FUSION_COMMON_ERROR_CHECK_H

#include <sstream>
#include <string>
#include <utility>

#include "Fusion/common/error/Error.h"
#include "Fusion/common/error/ErrorCode.h"

namespace fusion::error {

constexpr ErrorCode common_error(ErrorCategory category) noexcept {
   return ErrorCode{ErrorDomain::Common, category, 0};
}

namespace detail {

template <typename... Args>
std::string make_message(Args&&... args) {
   std::ostringstream oss;
   (oss << ... << std::forward<Args>(args));
   return oss.str();
}

} // namespace detail

} // namespace fusion::error

#define FUSION_THROW_CODE(code, ...)                                           \
   do {                                                                       \
      fusion::error::throw_error(                                           \
          (code),                                                             \
          fusion::error::detail::make_message(__VA_ARGS__),                 \
          std::source_location::current());                                   \
   } while (false)

#define FUSION_CHECK_CODE(cond, code, ...)                                     \
   do {                                                                       \
      if (!(static_cast<bool>(cond))) {                                        \
         FUSION_THROW_CODE((code), __VA_ARGS__);                              \
      }                                                                       \
   } while (false)

#define FUSION_CHECK(cond, ...)                                                \
   FUSION_CHECK_CODE(                                                          \
       (cond),                                                                \
       fusion::error::common_error(                                         \
           fusion::error::ErrorCategory::InvalidArgument),                  \
       __VA_ARGS__)

#define FUSION_CHECK_PRECONDITION(cond, ...)                                   \
   FUSION_CHECK_CODE(                                                          \
       (cond),                                                                \
       fusion::error::common_error(                                         \
           fusion::error::ErrorCategory::FailedPrecondition),               \
       __VA_ARGS__)

#define FUSION_CHECK_UNSUPPORTED(cond, ...)                                    \
   FUSION_CHECK_CODE(                                                          \
       (cond),                                                                \
       fusion::error::common_error(                                         \
           fusion::error::ErrorCategory::Unsupported),                      \
       __VA_ARGS__)

#define FUSION_CHECK_UNAVAILABLE(cond, ...)                                    \
   FUSION_CHECK_CODE(                                                          \
       (cond),                                                                \
       fusion::error::common_error(                                         \
           fusion::error::ErrorCategory::Unavailable),                      \
       __VA_ARGS__)

#define FUSION_INTERNAL_ASSERT(cond, ...)                                      \
   FUSION_CHECK_CODE(                                                          \
       (cond),                                                                \
       fusion::error::common_error(                                         \
           fusion::error::ErrorCategory::Internal),                         \
       __VA_ARGS__)

#endif // FUSION_COMMON_ERROR_CHECK_H