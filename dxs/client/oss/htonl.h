/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_OSS_HTONL_H_
#define DXS_CLIENT_OSS_HTONL_H_

// These are fast (f) versions of the hton* functions from netinet/in.h.

#include <cstdint>

#include "absl/base/config.h"
#include "absl/numeric/bits.h"

#ifdef ABSL_IS_LITTLE_ENDIAN

inline uint16_t fhtons(uint16_t x) { return absl::byteswap(x); }
inline uint32_t fhtonl(uint32_t x) { return absl::byteswap(x); }
inline uint64_t fhtonll(uint64_t x) { return absl::byteswap(x); }

#elif defined ABSL_IS_BIG_ENDIAN

inline uint16 fhtons(uint16 x) { return x; }
inline uint32 fhtonl(uint32 x) { return x; }
inline uint64 fhtonll(uint64 x) { return x; }

#else
#error \
    "Unsupported bytesex: Either ABSL_IS_BIG_ENDIAN or ABSL_IS_LITTLE_ENDIAN must be defined"  // NOLINT
#endif  // bytesex

// Inverse functions
inline uint16_t fntohs(uint16_t x) { return fhtons(x); }
inline uint32_t fntohl(uint32_t x) { return fhtonl(x); }
inline uint64_t fntohll(uint64_t x) { return fhtonll(x); }

#endif  // DXS_CLIENT_OSS_HTONL_H_
