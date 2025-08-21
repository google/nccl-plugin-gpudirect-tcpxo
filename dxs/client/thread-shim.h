/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_THREAD_SHIM_H_
#define DXS_CLIENT_THREAD_SHIM_H_

#include <memory>

#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"

namespace dxs {

// A holder of a thread that blocks until the thread terminates on destruction.
class ThreadShim {
 public:
  ThreadShim() = default;
  virtual ~ThreadShim() = default;
  ThreadShim(const ThreadShim&) = delete;
  ThreadShim& operator=(const ThreadShim&) = delete;
};

// Construct a new ThreadShim.
std::unique_ptr<ThreadShim> NewThreadShim(absl::AnyInvocable<void() &&> cb,
                                          absl::string_view thread_name);

}  // namespace dxs

#endif  // DXS_CLIENT_THREAD_SHIM_H_
