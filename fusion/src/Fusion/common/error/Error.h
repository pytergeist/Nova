#ifndef FUSION_COMMON_ERROR_H
#define FUSION_COMMON_ERROR_H

#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

#include "Fusion/common/error/ErrorCode.h"

namespace fusion::error {
class FusionError : public std::runtime_error {
 public:
   FusionError(
       const ErrorCode code, std::string message,
       const std::source_location location = std::source_location::current())
       : std::runtime_error(std::move(message)), code_(code),
         location_(location) {};

   [[nodiscard]] ErrorCode code() const noexcept { return code_; }

   [[nodiscard]] const std::source_location &location() const noexcept {
      return location_;
   }

 private:
   ErrorCode code_;
   std::source_location location_;
};

[[noreturn]] inline void throw_error(const ErrorCode code, std::string message,
                                     const std::source_location location) {
   throw FusionError(code, std::move(message), location);
}

} // namespace fusion::error

#endif // FUSION_COMMON_ERROR_H