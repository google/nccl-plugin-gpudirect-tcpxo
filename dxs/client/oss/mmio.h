/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_OSS_MMIO_H_
#define DXS_CLIENT_OSS_MMIO_H_

#include <atomic>
#include <cstdint>

#include "dxs/client/oss/barrier.h"

namespace platforms_util {

#if defined(__x86_64__)

inline void MmioWriteRelaxed8(volatile uint8_t* addr, uint8_t val) {
  __asm__ __volatile__("movb %0, %1" : : "q"(val), "m"(*addr) : "memory");
}

inline void MmioWriteRelaxed16(volatile uint16_t* addr, uint16_t val) {
  __asm__ __volatile__("movw %0, %1" : : "r"(val), "m"(*addr) : "memory");
}

inline void MmioWriteRelaxed32(volatile uint32_t* addr, uint32_t val) {
  __asm__ __volatile__("movl %0, %1" : : "r"(val), "m"(*addr) : "memory");
}

inline void MmioWriteRelaxed64(volatile uint64_t* addr, uint64_t val) {
  __asm__ __volatile__("movq %0, %1" : : "r"(val), "m"(*addr) : "memory");
}

// x86 provides relatively strong memory ordering by default, so we don't need
// explicit hardware barriers for these operations.
inline void MmioWriteRelease8(volatile uint8_t* addr, uint8_t val) {
  MmioWriteRelaxed8(addr, val);
}

inline void MmioWriteRelease16(volatile uint16_t* addr, uint16_t val) {
  MmioWriteRelaxed16(addr, val);
}

inline void MmioWriteRelease32(volatile uint32_t* addr, uint32_t val) {
  MmioWriteRelaxed32(addr, val);
}

inline void MmioWriteRelease64(volatile uint64_t* addr, uint64_t val) {
  MmioWriteRelaxed64(addr, val);
}

inline uint8_t MmioReadRelaxed8(const volatile uint8_t* addr) {
  uint8_t ret;
  __asm__ __volatile__("movb %1, %0" : "=q"(ret) : "m"(*addr) : "memory");
  return ret;
}

inline uint16_t MmioReadRelaxed16(const volatile uint16_t* addr) {
  uint16_t ret;
  __asm__ __volatile__("movw %1, %0" : "=r"(ret) : "m"(*addr) : "memory");
  return ret;
}

inline uint32_t MmioReadRelaxed32(const volatile uint32_t* addr) {
  uint32_t ret;
  __asm__ __volatile__("movl %1, %0" : "=r"(ret) : "m"(*addr) : "memory");
  return ret;
}

inline uint64_t MmioReadRelaxed64(const volatile uint64_t* addr) {
  uint64_t ret;
  __asm__ __volatile__("movq %1, %0" : "=r"(ret) : "m"(*addr) : "memory");
  return ret;
}

// x86 provides relatively strong memory ordering by default, so we don't need
// explicit hardware barriers for these operations.
inline uint8_t MmioReadAcquire8(const volatile uint8_t* addr) {
  auto ret = MmioReadRelaxed8(addr);
  return ret;
}

inline uint16_t MmioReadAcquire16(const volatile uint16_t* addr) {
  auto ret = MmioReadRelaxed16(addr);
  return ret;
}

inline uint32_t MmioReadAcquire32(const volatile uint32_t* addr) {
  auto ret = MmioReadRelaxed32(addr);
  return ret;
}

inline uint64_t MmioReadAcquire64(const volatile uint64_t* addr) {
  auto ret = MmioReadRelaxed64(addr);
  return ret;
}

#elif defined(__aarch64__)

inline void MmioWriteRelaxed8(volatile uint8_t* addr, uint8_t val) {
  __asm__ __volatile__("strb %w0, [%1]" : : "rZ"(val), "r"(addr) : "memory");
}

inline void MmioWriteRelaxed16(volatile uint16_t* addr, uint16_t val) {
  __asm__ __volatile__("strh %w0, [%1]" : : "rZ"(val), "r"(addr) : "memory");
}

inline void MmioWriteRelaxed32(volatile uint32_t* addr, uint32_t val) {
  __asm__ __volatile__("str %w0, [%1]" : : "rZ"(val), "r"(addr) : "memory");
}

inline void MmioWriteRelaxed64(volatile uint64_t* addr, uint64_t val) {
  __asm__ __volatile__("str %x0, [%1]" : : "rZ"(val), "r"(addr) : "memory");
}

// Aarch64 provides a weaker memory ordering by default, so we need explicit
// hardware barriers to ensure the ordering of these operations.
inline void MmioWriteRelease8(volatile uint8_t* addr, uint8_t val) {
  MmioWriteBarrier();
  MmioWriteRelaxed8(addr, val);
}

inline void MmioWriteRelease16(volatile uint16_t* addr, uint16_t val) {
  MmioWriteBarrier();
  MmioWriteRelaxed16(addr, val);
}

inline void MmioWriteRelease32(volatile uint32_t* addr, uint32_t val) {
  MmioWriteBarrier();
  MmioWriteRelaxed32(addr, val);
}

inline void MmioWriteRelease64(volatile uint64_t* addr, uint64_t val) {
  MmioWriteBarrier();
  MmioWriteRelaxed64(addr, val);
}

inline uint8_t MmioReadRelaxed8(const volatile uint8_t* addr) {
  uint8_t ret;
  __asm__ __volatile__("ldrb %w0, [%1]" : "=r"(ret) : "r"(addr) : "memory");
  return ret;
}

inline uint16_t MmioReadRelaxed16(const volatile uint16_t* addr) {
  uint16_t ret;
  __asm__ __volatile__("ldrh %w0, [%1]" : "=r"(ret) : "r"(addr) : "memory");
  return ret;
}

inline uint32_t MmioReadRelaxed32(const volatile uint32_t* addr) {
  uint32_t ret;
  __asm__ __volatile__("ldr %w0, [%1]" : "=r"(ret) : "r"(addr) : "memory");
  return ret;
}

inline uint64_t MmioReadRelaxed64(const volatile uint64_t* addr) {
  uint64_t ret;
  __asm__ __volatile__("ldr %0, [%1]" : "=r"(ret) : "r"(addr) : "memory");
  return ret;
}

// Aarch64 provides a weaker memory ordering by default, so we need explicit
// hardware barriers to ensure the ordering of these operations.
inline uint8_t MmioReadAcquire8(const volatile uint8_t* addr) {
  uint8_t ret = MmioReadRelaxed8(addr);
  MmioReadBarrier();
  return ret;
}

inline uint16_t MmioReadAcquire16(const volatile uint16_t* addr) {
  uint16_t ret = MmioReadRelaxed16(addr);
  MmioReadBarrier();
  return ret;
}

inline uint32_t MmioReadAcquire32(const volatile uint32_t* addr) {
  uint32_t ret = MmioReadRelaxed32(addr);
  MmioReadBarrier();
  return ret;
}

inline uint64_t MmioReadAcquire64(const volatile uint64_t* addr) {
  uint64_t ret = MmioReadRelaxed64(addr);
  MmioReadBarrier();
  return ret;
}

#elif defined(__riscv)

inline void MmioWriteRelaxed8(volatile uint8_t* addr, uint8_t val) {
  __asm__ __volatile__("sb %0, 0(%1)" : : "r"(val), "r"(addr) : "memory");
}

inline void MmioWriteRelaxed16(volatile uint16_t* addr, uint16_t val) {
  __asm__ __volatile__("sh %0, 0(%1)" : : "r"(val), "r"(addr) : "memory");
}

inline void MmioWriteRelaxed32(volatile uint32_t* addr, uint32_t val) {
  __asm__ __volatile__("sw %0, 0(%1)" : : "r"(val), "r"(addr) : "memory");
}

inline void MmioWriteRelaxed64(volatile uint64_t* addr, uint64_t val) {
  __asm__ __volatile__("sd %0, 0(%1)" : : "r"(val), "r"(addr) : "memory");
}

// RISC-V provides a weaker memory ordering by default, so we need explicit
// hardware barriers to ensure the ordering of these operations.
inline void MmioWriteRelease8(volatile uint8_t* addr, uint8_t val) {
  MmioWriteBarrier();
  MmioWriteRelaxed8(addr, val);
}

inline void MmioWriteRelease16(volatile uint16_t* addr, uint16_t val) {
  MmioWriteBarrier();
  MmioWriteRelaxed16(addr, val);
}

inline void MmioWriteRelease32(volatile uint32_t* addr, uint32_t val) {
  MmioWriteBarrier();
  MmioWriteRelaxed32(addr, val);
}

inline void MmioWriteRelease64(volatile uint64_t* addr, uint64_t val) {
  MmioWriteBarrier();
  MmioWriteRelaxed64(addr, val);
}

inline uint8_t MmioReadRelaxed8(const volatile uint8_t* addr) {
  uint8_t ret;
  __asm__ __volatile__("lb %0, 0(%1)" : "=r"(ret) : "r"(addr) : "memory");
  return ret;
}

inline uint16_t MmioReadRelaxed16(const volatile uint16_t* addr) {
  uint16_t ret;
  __asm__ __volatile__("lh %0, 0(%1)" : "=r"(ret) : "r"(addr) : "memory");
  return ret;
}

inline uint32_t MmioReadRelaxed32(const volatile uint32_t* addr) {
  uint32_t ret;
  __asm__ __volatile__("lw %0, 0(%1)" : "=r"(ret) : "r"(addr) : "memory");
  return ret;
}

inline uint64_t MmioReadRelaxed64(const volatile uint64_t* addr) {
  uint64_t ret;
  __asm__ __volatile__("ld %0, 0(%1)" : "=r"(ret) : "r"(addr) : "memory");
  return ret;
}

// RISC-V provides a weaker memory ordering by default, so we need explicit
// hardware barriers to ensure the ordering of these operations.
inline uint8_t MmioReadAcquire8(const volatile uint8_t* addr) {
  uint8_t ret = MmioReadRelaxed8(addr);
  MmioReadBarrier();
  return ret;
}

inline uint16_t MmioReadAcquire16(const volatile uint16_t* addr) {
  uint16_t ret = MmioReadRelaxed16(addr);
  MmioReadBarrier();
  return ret;
}

inline uint32_t MmioReadAcquire32(const volatile uint32_t* addr) {
  uint32_t ret = MmioReadRelaxed32(addr);
  MmioReadBarrier();
  return ret;
}

inline uint64_t MmioReadAcquire64(const volatile uint64_t* addr) {
  uint64_t ret = MmioReadRelaxed64(addr);
  MmioReadBarrier();
  return ret;
}

#else

namespace internal {

// Avoids compiler reordering relative to this call.
void CompilerBarrier() { __asm__ __volatile__("" : : : "memory"); }

template <typename T>
void MmioWrite(volatile T* addr, T val) {
  CompilerBarrier();
#if defined(THREAD_SANITIZER)
  using AtomicT = std::atomic<T>;

  static_assert(AtomicT::is_always_lock_free);
  static_assert(sizeof(AtomicT) == sizeof(T));
  static_assert(alignof(AtomicT) == alignof(T));

  reinterpret_cast<volatile AtomicT*>(addr)->store(val,
                                                   std::memory_order_release);
#else
  *addr = val;
#endif
  CompilerBarrier();
}

template <typename T>
T MmioRead(const volatile T* addr) {
  T ret;

  CompilerBarrier();
#if defined(THREAD_SANITIZER)
  using AtomicT = std::atomic<T>;

  static_assert(AtomicT::is_always_lock_free);
  static_assert(sizeof(AtomicT) == sizeof(T));
  static_assert(alignof(AtomicT) == alignof(T));

  ret = reinterpret_cast<const volatile AtomicT*>(addr)->load(
      std::memory_order_acquire);
#else
  ret = *addr;
#endif
  CompilerBarrier();

  return ret;
}

}  // namespace internal

inline void MmioWriteRelaxed8(volatile uint8_t* addr, uint8_t val) {
  return internal::MmioWrite(addr, val);
}

inline void MmioWriteRelaxed16(volatile uint16_t* addr, uint16_t val) {
  return internal::MmioWrite(addr, val);
}

inline void MmioWriteRelaxed32(volatile uint32_t* addr, uint32_t val) {
  return internal::MmioWrite(addr, val);
}

inline void MmioWriteRelaxed64(volatile uint64_t* addr, uint64_t val) {
  return internal::MmioWrite(addr, val);
}

inline void MmioWriteRelease8(volatile uint8_t* addr, uint8_t val) {
  return internal::MmioWrite(addr, val);
}

inline void MmioWriteRelease16(volatile uint16_t* addr, uint16_t val) {
  return internal::MmioWrite(addr, val);
}

inline void MmioWriteRelease32(volatile uint32_t* addr, uint32_t val) {
  return internal::MmioWrite(addr, val);
}

inline void MmioWriteRelease64(volatile uint64_t* addr, uint64_t val) {
  return internal::MmioWrite(addr, val);
}

inline uint8_t MmioReadRelaxed8(const volatile uint8_t* addr) {
  return internal::MmioRead(addr);
}

inline uint16_t MmioReadRelaxed16(const volatile uint16_t* addr) {
  return internal::MmioRead(addr);
}

inline uint32_t MmioReadRelaxed32(const volatile uint32_t* addr) {
  return internal::MmioRead(addr);
}

inline uint64_t MmioReadRelaxed64(const volatile uint64_t* addr) {
  return internal::MmioRead(addr);
}

inline uint8_t MmioReadAcquire8(const volatile uint8_t* addr) {
  return internal::MmioRead(addr);
}

inline uint16_t MmioReadAcquire16(const volatile uint16_t* addr) {
  return internal::MmioRead(addr);
}

inline uint32_t MmioReadAcquire32(const volatile uint32_t* addr) {
  return internal::MmioRead(addr);
}

inline uint64_t MmioReadAcquire64(const volatile uint64_t* addr) {
  return internal::MmioRead(addr);
}

#endif

inline void MmioWriteRelaxed(volatile uint8_t* addr, uint8_t val) {
  MmioWriteRelaxed8(addr, val);
}

inline void MmioWriteRelaxed(volatile uint16_t* addr, uint16_t val) {
  MmioWriteRelaxed16(addr, val);
}

inline void MmioWriteRelaxed(volatile uint32_t* addr, uint32_t val) {
  MmioWriteRelaxed32(addr, val);
}

inline void MmioWriteRelaxed(volatile uint64_t* addr, uint64_t val) {
  MmioWriteRelaxed64(addr, val);
}

inline void MmioWriteRelease(volatile uint8_t* addr, uint8_t val) {
  MmioWriteRelease8(addr, val);
}

inline void MmioWriteRelease(volatile uint16_t* addr, uint16_t val) {
  MmioWriteRelease16(addr, val);
}

inline void MmioWriteRelease(volatile uint32_t* addr, uint32_t val) {
  MmioWriteRelease32(addr, val);
}

inline void MmioWriteRelease(volatile uint64_t* addr, uint64_t val) {
  MmioWriteRelease64(addr, val);
}

inline uint8_t MmioReadRelaxed(const volatile uint8_t* addr) {
  return MmioReadRelaxed8(addr);
}

inline uint16_t MmioReadRelaxed(const volatile uint16_t* addr) {
  return MmioReadRelaxed16(addr);
}

inline uint32_t MmioReadRelaxed(const volatile uint32_t* addr) {
  return MmioReadRelaxed32(addr);
}

inline uint64_t MmioReadRelaxed(const volatile uint64_t* addr) {
  return MmioReadRelaxed64(addr);
}

inline uint8_t MmioReadAcquire(const volatile uint8_t* addr) {
  return MmioReadAcquire8(addr);
}

inline uint16_t MmioReadAcquire(const volatile uint16_t* addr) {
  return MmioReadAcquire16(addr);
}

inline uint32_t MmioReadAcquire(const volatile uint32_t* addr) {
  return MmioReadAcquire32(addr);
}

inline uint64_t MmioReadAcquire(const volatile uint64_t* addr) {
  return MmioReadAcquire64(addr);
}

}  // namespace platforms_util

#endif  // DXS_CLIENT_OSS_MMIO_H_
