/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_OSS_BARRIER_H_
#define DXS_CLIENT_OSS_BARRIER_H_

namespace platforms_util {

#if defined(__x86_64__)
inline void DmaReadBarrier() { __asm__ __volatile__("lfence" : : : "memory"); }

inline void DmaWriteBarrier() { __asm__ __volatile__("sfence" : : : "memory"); }

inline void DmaFullBarrier() { __asm__ __volatile__("mfence" : : : "memory"); }

inline void MmioReadBarrier() { __asm__ __volatile__("lfence" : : : "memory"); }

inline void MmioWriteBarrier() {
  __asm__ __volatile__("sfence" : : : "memory");
}

inline void MmioFullBarrier() { __asm__ __volatile__("mfence" : : : "memory"); }
#elif defined(__powerpc__)
inline void DmaReadBarrier() { __asm__ __volatile__("lwsync" : : : "memory"); }

inline void DmaWriteBarrier() { __asm__ __volatile__("lwsync" : : : "memory"); }

inline void DmaFullBarrier() { __asm__ __volatile__("lwsync" : : : "memory"); }

inline void MmioReadBarrier() { __asm__ __volatile__("sync" : : : "memory"); }

inline void MmioWriteBarrier() { __asm__ __volatile__("sync" : : : "memory"); }

inline void MmioFullBarrier() { __asm__ __volatile__("sync" : : : "memory"); }
#elif defined(__aarch64__)
inline void DmaReadBarrier() {
  __asm__ __volatile__("dmb oshld" : : : "memory");
}

inline void DmaWriteBarrier() {
  __asm__ __volatile__("dmb oshst" : : : "memory");
}

inline void DmaFullBarrier() { __asm__ __volatile__("dmb osh" : : : "memory"); }

inline void MmioReadBarrier() { __asm__ __volatile__("dmb ld" : : : "memory"); }

inline void MmioWriteBarrier() {
  __asm__ __volatile__("dmb st" : : : "memory");
}

inline void MmioFullBarrier() { __asm__ __volatile__("dmb sy" : : : "memory"); }
#elif defined(__riscv)
// These barriers do not need to enforce ordering on devices, just memory.
inline void DmaReadBarrier() {
  __asm__ __volatile__("fence r, rw" : : : "memory");
}

inline void DmaWriteBarrier() {
  __asm__ __volatile__("fence w, w" : : : "memory");
}

inline void DmaFullBarrier() {
  __asm__ __volatile__("fence rw, rw" : : : "memory");
}

// These barriers need to enforce ordering on both devices and memory.
inline void MmioReadBarrier() {
  __asm__ __volatile__("fence ir, iorw" : : : "memory");
}

inline void MmioWriteBarrier() {
  __asm__ __volatile__("fence ow, ow" : : : "memory");
}

inline void MmioFullBarrier() {
  __asm__ __volatile__("fence iorw, iorw" : : : "memory");
}
#else
#error "Missing device memory barrier implementations for this architecture."
#endif

}  // namespace platforms_util

#endif  // DXS_CLIENT_OSS_BARRIER_H_
