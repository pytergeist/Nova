// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#include "ThreadPool.h"

#include <stdexcept>

JoinThreads::JoinThreads(std::vector<std::thread> &threads)
    : threads_(threads) {}

JoinThreads::~JoinThreads() {
   for (auto &t : threads_) {
      if (t.joinable()) {
         t.join();
      }
   }
}

ThreadPool::ThreadPool() : done_(false), join_threads_(threads_) {

   const unsigned thread_count = std::thread::hardware_concurrency();

   try {
      for (unsigned i = 0; i < thread_count; ++i) {
         threads_.emplace_back(&ThreadPool::worker_thread, this);
      }
   } catch (...) {
      done_ = true;
      throw;
   }
}

ThreadPool::~ThreadPool() { done_ = true; }

void ThreadPool::submit(std::function<void()> task) {
   queue_.push(std::move(task));
}

void ThreadPool::worker_thread() {
   while (!done_) {
      std::function<void()> task;
      if (queue_.try_pop(task)) {
         task();
      } else {
         std::this_thread::yield();
      }
   }
}
