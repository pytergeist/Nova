#pragma once
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace fusionlog {

enum class Level : int { kError = 0, kWarn = 1, kInfo = 2, kDebug = 3 };

// Compile-time max level (higher = more logs compiled in)
// Change default to kDebug if you want more by default.
#ifndef FUSION_LOG_COMPILED_LEVEL
#define FUSION_LOG_COMPILED_LEVEL 2 // 0=E,1=W,2=I,3=D
#endif

// Optional: runtime level (default = compiled level)
// Set via env var FUSION_LOG_LEVEL (0..3). We cache after first read.
inline Level runtime_level() {
   static Level cached = [] {
      const char *e = std::getenv("FUSION_LOG_LEVEL");
      if (!e)
         return static_cast<Level>(FUSION_LOG_COMPILED_LEVEL);
      int v = std::atoi(e);
      if (v < 0)
         v = 0;
      if (v > 3)
         v = 3;
      return static_cast<Level>(v);
   }();
   return cached;
}

inline const char *level_tag(Level L) {
   switch (L) {
   case Level::kError:
      return "E";
   case Level::kWarn:
      return "W";
   case Level::kInfo:
      return "I";
   case Level::kDebug:
      return "D";
   }
   return "?";
}

inline std::mutex &log_mutex() {
   static std::mutex m;
   return m;
}

template <typename... Args>
inline void log_line(Level L, const char *file, int line, Args &&...args) {
#if FUSION_LOG_COMPILED_LEVEL >= 0
   // If compiled out below the level, drop at compile time:
   if (static_cast<int>(L) > FUSION_LOG_COMPILED_LEVEL)
      return;
   // If runtime level is lower, drop at runtime:
   if (static_cast<int>(L) > static_cast<int>(runtime_level()))
      return;

   std::ostringstream oss;

   // Timestamp
   using clock = std::chrono::system_clock;
   const auto now = clock::now();
   const auto tt = clock::to_time_t(now);
   const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                       now.time_since_epoch()) %
                   1000;

   std::tm tm_buf;
#if defined(_WIN32)
   localtime_s(&tm_buf, &tt);
#else
   localtime_r(&tt, &tm_buf);
#endif

   oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << '.'
       << std::setfill('0') << std::setw(3) << ms.count();

   // Thread id (shortened)
   std::ostringstream tid;
   tid << std::this_thread::get_id();

   oss << " [" << level_tag(L) << "] " << file << ":" << line
       << " (tid=" << tid.str() << ") ";

   // Variadic message
   (void)std::initializer_list<int>{((oss << std::forward<Args>(args)), 0)...};

   // Emit
   std::lock_guard<std::mutex> lk(log_mutex());
   std::cerr << oss.str() << '\n';
#endif
}

} // namespace fusionlog

#define FUSION_LOGE(...)                                                       \
   ::fusionlog::log_line(::fusionlog::Level::kError, __FILE__, __LINE__,       \
                         __VA_ARGS__)
#define FUSION_LOGW(...)                                                       \
   ::fusionlog::log_line(::fusionlog::Level::kWarn, __FILE__, __LINE__,        \
                         __VA_ARGS__)
#define FUSION_LOGI(...)                                                       \
   ::fusionlog::log_line(::fusionlog::Level::kInfo, __FILE__, __LINE__,        \
                         __VA_ARGS__)
#define FUSION_LOGD(...)                                                       \
   ::fusionlog::log_line(::fusionlog::Level::kDebug, __FILE__, __LINE__,       \
                         __VA_ARGS__)
