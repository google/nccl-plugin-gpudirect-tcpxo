/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_BASE_INTERFACE_H_
#define DXS_CLIENT_BASE_INTERFACE_H_

namespace dxs {

// A non-copyable non-movable protected-constructible interface to remove
// boilerplate from writing interface classes.
class BaseInterface {
 protected:
  BaseInterface() = default;

 public:
  virtual ~BaseInterface() = default;
  BaseInterface(const BaseInterface&) = delete;
  BaseInterface& operator=(const BaseInterface&) = delete;
  BaseInterface(BaseInterface&&) = delete;
  BaseInterface& operator=(BaseInterface&&) = delete;
};

}  // namespace dxs

#endif  // DXS_CLIENT_BASE_INTERFACE_H_
