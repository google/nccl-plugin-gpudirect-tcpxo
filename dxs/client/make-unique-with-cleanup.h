/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_MAKE_UNIQUE_WITH_CLEANUP_H_
#define DXS_CLIENT_MAKE_UNIQUE_WITH_CLEANUP_H_

#include <memory>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/memory/memory.h"

namespace dxs {
// Implementation details
namespace make_unique_with_cleanup_internal {

template <class Base, class Functor>
class DestructorWrapper : public Base {
 public:
  template <class... Args>
  explicit DestructorWrapper(Functor cleanup, Args&&... args)
      : Base(std::forward<Args>(args)...), cleanup_(std::move(cleanup)) {}
  ~DestructorWrapper() override { std::move(cleanup_)(); }
  // DestructorWrapper is neither copyable nor movable.
  DestructorWrapper(const DestructorWrapper&) = delete;
  DestructorWrapper& operator=(const DestructorWrapper&) = delete;

 private:
  ABSL_ATTRIBUTE_NO_UNIQUE_ADDRESS Functor cleanup_;
};

}  // namespace make_unique_with_cleanup_internal

// MakeUniqueWithCleanup<T> creates a unique_ptr<T> that binds a cleanup functor
// to run on object destruction. It is useful to attach RAII pre-destructor
// behavior to interfaces without needing a bespoke implementation.
template <class Base, int&... ExplicitParameterBarrier, class Functor,
          class... Args>
std::unique_ptr<Base> MakeUniqueWithCleanup(Functor&& cleanup, Args&&... args) {
  static_assert(std::has_virtual_destructor_v<Base>,
                "MakeUniqueWithCleanup requires a virtual destructor.");
  static_assert(!std::is_final_v<Base>,
                "MakeUniqueWithCleanup does not work for final classes.");
  return absl::WrapUnique(
      new make_unique_with_cleanup_internal::DestructorWrapper<
          Base, std::decay_t<Functor>>(std::forward<Functor>(cleanup),
                                       std::forward<Args>(args)...));
}

}  // namespace dxs

#endif  // DXS_CLIENT_MAKE_UNIQUE_WITH_CLEANUP_H_
