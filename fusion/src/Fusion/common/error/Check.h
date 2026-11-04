#ifndef FUSION_COMMON_ERROR_CHECK_H
#define FUSION_COMMON_ERROR_CHECK_H

#include <sstream>
#include <string>
#include <utility>

#include "Fusion/common/error/Error.h"
#include "Fusion/common/error/ErrorCode.h"

namespace fusion::error {

constexpr ErrorCode common_error(const ErrorCategory category) noexcept {
   return ErrorCode{
       .domain = ErrorDomain::Common, .category = category, .detail = 0};
}

namespace detail {

template <typename... Args> std::string make_message(Args &&...args) {
   std::ostringstream oss;
   (oss << ... << std::forward<Args>(args));
   return oss.str();
}

} // namespace detail

template <typename... Args> std::string message(Args &&...args) {
   return detail::make_message(std::forward<Args>(args)...);
}

} // namespace fusion::error

#define FUSION_THROW_CODE(code, message_expr)                                  \
   ::fusion::error::throw_error((code), (message_expr),                        \
                                std::source_location::current())

#define FUSION_CHECK_CODE(cond, code, message_expr)                            \
   (static_cast<bool>(cond)                                                    \
        ? static_cast<void>(0)                                                 \
        : ::fusion::error::throw_error((code), (message_expr),                 \
                                       std::source_location::current()))

#define FUSION_CHECK(cond, message_expr)                                       \
   FUSION_CHECK_CODE((cond),                                                   \
                     ::fusion::error::common_error(                            \
                         ::fusion::error::ErrorCategory::InvalidArgument),     \
                     (message_expr))

#define FUSION_CHECK_PRECONDITION(cond, message_expr)                          \
   FUSION_CHECK_CODE((cond),                                                   \
                     ::fusion::error::common_error(                            \
                         ::fusion::error::ErrorCategory::FailedPrecondition),  \
                     (message_expr))

#define FUSION_CHECK_UNSUPPORTED(cond, message_expr)                           \
   FUSION_CHECK_CODE((cond),                                                   \
                     ::fusion::error::common_error(                            \
                         ::fusion::error::ErrorCategory::Unsupported),         \
                     (message_expr))

#define FUSION_CHECK_UNAVAILABLE(cond, message_expr)                           \
   FUSION_CHECK_CODE((cond),                                                   \
                     ::fusion::error::common_error(                            \
                         ::fusion::error::ErrorCategory::Unavailable),         \
                     (message_expr))

#define FUSION_INTERNAL_ASSERT(cond, message_expr)                             \
   FUSION_CHECK_CODE((cond),                                                   \
                     ::fusion::error::common_error(                            \
                         ::fusion::error::ErrorCategory::Internal),            \
                     (message_expr))

#define FUSION_INTERNAL_ASSERT_CODE(cond, code, message_expr)                  \
   FUSION_CHECK_CODE((cond), (code), (message_expr))

#endif // FUSION_COMMON_ERROR_CHECK_H