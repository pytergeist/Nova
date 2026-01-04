// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef THREAD_SAFE_QUEUE_HPP
#define THREAD_SAFE_QUEUE_HPP

#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T> class ThreadSafeQueue {
   mutable std::mutex mutex_;
   std::queue<T> queue_;
   std::condition_variable condition_;

 public:
   ThreadSafeQueue() = default;

   void push(const T &value) {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(value);
      condition_.notify_one();
   }

   std::shared_ptr<T> wait_and_pop() {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [this] { return !queue_.empty(); });
      std::shared_ptr<T> res = std::make_shared<T>(std::move(queue_.front()));
      queue_.pop();
      return res;
   }

   bool try_pop(T &value) {
      std::unique_lock<std::mutex> lock(mutex_);
      if (queue_.empty()) {
         return false;
      }
      value = std::move(queue_.front());
      queue_.pop();
      return true;
   }

   std::shared_ptr<T> try_pop() {
      // shared_ptr implementation of try_pop
      std::unique_lock<std::mutex> lock(mutex_);
      if (queue_.empty()) {
         return std::shared_ptr<T>();
      };
      std::shared_ptr<T> res = std::make_shared<T>(std::move(queue_.front()));
      queue_.pop();
      return res;
   }

   bool empty() const {
      std::lock_guard<std::mutex> lock(mutex_);
      return queue_.empty();
   }
};

#endif // THREAD_SAFE_QUEUE_HPP
