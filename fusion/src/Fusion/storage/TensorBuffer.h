#ifndef TENSOR_BUFFER_H
#define TENSOR_BUFFER_H

#include <cstddef>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <memory>
#include <vector>
#include <cstring>

inline void *aligned_alloc_bytes(size_t alignment, size_t size) {
  if (alignment < alignof(void *) || (alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument(
        "alignment must be a power of two and >= alignof(void*)");
  }

  void *p = nullptr;
  int rc = posix_memalign(&p, alignment, size);
  if (rc != 0 || !p) {
    throw std::bad_alloc();
  }
  return p;
}

struct AlignedFree {
  inline void operator()(void *ptr) const noexcept { std::free(ptr); }
};

class TensorBuffer {
public:
  TensorBuffer() = default;

  static TensorBuffer allocate(size_t size_bytes, size_t alignment = 64) {
    if (size_bytes == 0)
      return {};
    void *p = aligned_alloc_bytes(alignment, size_bytes);
    return TensorBuffer(p, size_bytes, alignment);
  };

  template <typename T>
  static TensorBuffer allocate_elements(std::size_t count,
                                        std::size_t alignment = 64) {
    return TensorBuffer::allocate(count * sizeof(T), alignment);
  }

  template <typename T>
  T* data_as(std::size_t byte_off = 0) noexcept {
    return reinterpret_cast<T *>(static_cast<std::byte *>(ptr_.get()) + byte_off);
}

  template <typename T>
  const T* data_as(std::size_t byte_off = 0) const noexcept {
    return reinterpret_cast<const T*>(static_cast<const std::byte*>(ptr_.get()) + byte_off);
  }


  TensorBuffer(const TensorBuffer &) = default;
  TensorBuffer &operator=(const TensorBuffer &) = default;
  TensorBuffer(TensorBuffer &&) noexcept = default;
  TensorBuffer &operator=(TensorBuffer &&) noexcept = default;

  void *data() noexcept { return ptr_.get(); };
  const void *data() const noexcept { return ptr_.get(); };

  template <typename T>
  T *data() noexcept { return reinterpret_cast<T *>(static_cast<std::byte *>(ptr_.get())); }

  template <typename T>
  const T *data() const noexcept { return reinterpret_cast<T *>(static_cast<std::byte *>(ptr_.get())); }

  template <typename T>
  T* data_ptr(std::size_t elem_off = 0) noexcept {
    return data_as<T>(elem_off * sizeof(T));
  }

  template <typename T>
  const T* data_ptr(std::size_t elem_off = 0) const noexcept {
    return data_as<const T>(elem_off * sizeof(T));
  }

  template <typename T> std::size_t size() const noexcept {return size_ / sizeof(T);};
  std::size_t size_bytes() const noexcept { return size_; };
  bool empty() const noexcept { return size_ == 0; };
  std::size_t alignment() const noexcept { return alignment_; };
  explicit operator bool() const noexcept { return ptr_ != nullptr; };
  std::size_t use_count() const noexcept { return ptr_.use_count(); }

  template <typename T>
  void copy_from(const std::vector<T>& src, std::size_t dst_elem_offset = 0) {
    if (size_bytes() == 0 || src.empty())
      throw std::out_of_range("copy_from: dst buffer or src is empty");

    const std::size_t needed_bytes = (dst_elem_offset + src.size()) * sizeof(T);
    if (needed_bytes > size_bytes())
      throw std::out_of_range("copy_from: dst buffer too small for copy (with offset)");

    std::memcpy(data_ptr<T>(dst_elem_offset), src.data(), src.size() * sizeof(T));
  }

private:
  std::shared_ptr<void> ptr_{};
  size_t size_{0};
  size_t alignment_{alignof(std::max_align_t)};

  TensorBuffer(void *raw, size_t size, size_t alignment)
      : ptr_(raw, AlignedFree{}), size_(size), alignment_(alignment) {}
};
#endif // TENSOR_BUFFER_H
