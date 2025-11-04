#ifndef CHECKS_H
#define CHECKS_H
#pragma once

#include <stdexcept>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <source_location>  // C++20

namespace fusion::detail {

// Emit a uniform error message.
inline std::string make_msg(std::string_view head,
                            std::string_view body,
                            std::source_location loc = std::source_location::current()) {
  std::ostringstream oss;
  oss << head << " @ " << loc.file_name() << ':' << loc.line();
  if (!body.empty()) oss << ": " << body;
  return std::move(oss).str();
}

inline void check(bool cond,
                  std::string_view msg = {},
                  std::source_location loc = std::source_location::current()) {
  if (!cond) {
    throw std::runtime_error(make_msg("FUSION_CHECK failed", msg, loc));
  }
}

inline void bounds_check(long long idx,
                         long long size,
                         std::source_location loc = std::source_location::current()) {
  if (idx < 0 || idx >= size) {
    std::ostringstream oss;
    oss << "Index " << idx << " out of bounds for size " << size;
    throw std::out_of_range(make_msg("FUSION_BOUNDS_CHECK failed", oss.str(), loc));
  }
}

// SFINAE: requires size() and ndims() members.
template <class T>
using has_size_and_ndims_t = std::conjunction<
  std::is_same<decltype(std::declval<const T&>().size()), std::size_t>,
  std::is_same<decltype(std::declval<const T&>().ndims()), std::size_t>
>;

template <class T1, class T2,
          std::enable_if_t<has_size_and_ndims_t<T1>::value &&
                           has_size_and_ndims_t<T2>::value, int> = 0>
inline void allow_scalar_binary(const T1& t1, const T2& t2,
                                std::source_location loc = std::source_location::current()) {
  const auto sz1 = static_cast<long long>(t1.size());
  const auto sz2 = static_cast<long long>(t2.size());
  const bool is_scalar1 = (t1.ndims() == 0) || (sz1 == 1);
  const bool is_scalar2 = (t2.ndims() == 0) || (sz2 == 1);
  if (!(is_scalar1 || is_scalar2 || (sz1 == sz2))) {
    std::ostringstream oss;
    oss << "Elementwise op requires scalars or equal element counts; got size("
        << sz1 << ") vs size(" << sz2 << ")";
    throw std::runtime_error(make_msg("FUSION_ALLOW_SCALAR_BINARY failed", oss.str(), loc));
  }
}

} // namespace fusion::detail


#define FUSION_CHECK(cond, msg) \
  ::fusion::detail::check(static_cast<bool>(cond), (msg))

#define FUSION_BOUNDS_CHECK(idx, size) \
  ::fusion::detail::bounds_check(static_cast<long long>(idx), static_cast<long long>(size))

#define FUSION_ALLOW_SCALAR_BINARY(t1, t2) \
  ::fusion::detail::allow_scalar_binary((t1), (t2))

#endif // CHECKS_H
