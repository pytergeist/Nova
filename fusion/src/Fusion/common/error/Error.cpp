#include "Fusion/common/error/Error.h"

namespace fusion::error {

[[noreturn]] void throw_error(const ErrorCode code,
                              std::string message,
                              const std::source_location location) {
   throw FusionError(code, std::move(message), location);
}

} // namespace fusion::error