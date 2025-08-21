/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"
#include "dxs/client/thread-shim.h"

namespace dxs {
namespace {

class ExternalThreadShim : public ThreadShim {
 public:
  explicit ExternalThreadShim(absl::AnyInvocable<void() &&> cb)
      : thread_(std::move(cb)) {}

  ~ExternalThreadShim() override { thread_.join(); }

 private:
  std::thread thread_;
};

}  // namespace

std::unique_ptr<ThreadShim> NewThreadShim(absl::AnyInvocable<void() &&> cb,
                                          absl::string_view thread_name) {
  return std::make_unique<ExternalThreadShim>(std::move(cb));
}

}  // namespace dxs
