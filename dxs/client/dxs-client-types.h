/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_DXS_CLIENT_TYPES_H_
#define DXS_CLIENT_DXS_CLIENT_TYPES_H_

#include <cstdint>

namespace dxs {

// Use enum class instead of typedef for stricter type checking.
enum class ListenSocketHandle : uint64_t {};
enum class DataSocketHandle : uint64_t {};

enum class CollectiveListenSocketHandle : uint64_t {};
enum class CollectiveDataSocketHandle : uint64_t {};

typedef uint64_t Reg;
typedef uint64_t OpId;

constexpr Reg kInvalidRegistration = 0ul;

constexpr char kDefaultDxsAddr[] = "169.254.169.254";
constexpr char kDefaultDxsPort[] = "55555";

constexpr uint16_t kBufferManagerSourcePort = 338;

constexpr int kMaxHostnameSize = 79;

}  // namespace dxs

#endif  // DXS_CLIENT_DXS_CLIENT_TYPES_H_
