/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef DXS_CLIENT_OSS_STATUS_MACROS_H_
#define DXS_CLIENT_OSS_STATUS_MACROS_H_

#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/log_severity.h"
#include "absl/base/optimization.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace dxs::status_macros_internal {

class ABSL_MUST_USE_RESULT StatusBuilder {
 public:
  explicit StatusBuilder(absl::Status original_status)
      : rep_(InitRep(std::move(original_status))) {}
  ~StatusBuilder() = default;
  StatusBuilder(StatusBuilder&&) = default;
  StatusBuilder& operator=(StatusBuilder&&) = default;

  // Mutates the builder so that the result status will be logged when this
  // builder is converted to a Status.
  StatusBuilder& Log(absl::LogSeverity level) & {
    if (!rep_) return *this;
    rep_->severity = level;
    return *this;
  }
  ABSL_MUST_USE_RESULT StatusBuilder&& Log(absl::LogSeverity level) && {
    return std::move(Log(level));
  }
  StatusBuilder& LogError() & { return Log(absl::LogSeverity::kError); }
  ABSL_MUST_USE_RESULT StatusBuilder&& LogError() && {
    return std::move(LogError());
  }
  StatusBuilder& LogWarning() & { return Log(absl::LogSeverity::kWarning); }
  ABSL_MUST_USE_RESULT StatusBuilder&& LogWarning() && {
    return std::move(LogWarning());
  }
  StatusBuilder& LogInfo() & { return Log(absl::LogSeverity::kInfo); }
  ABSL_MUST_USE_RESULT StatusBuilder&& LogInfo() && {
    return std::move(LogInfo());
  }

  // Appends to the extra message that will be added to the original status.  By
  // default, the extra message is added to the original message as if by
  // `util::Annotate`, which includes a convenience separator between the
  // original message and the enriched one.
  template <typename T>
  StatusBuilder& operator<<(const T& value) & {
    if (rep_) rep_->stream << value;
    return *this;
  }

  template <typename T>
  ABSL_MUST_USE_RESULT StatusBuilder&& operator<<(const T& value) && {
    if (rep_) rep_->stream << value;
    return std::move(*this);
  }

  template <typename Adaptor>
  auto With(Adaptor&& adaptor) {
    return std::forward<Adaptor>(adaptor)(std::move(*this));
  }

  operator absl::Status() {  // NOLINT: Builder converts implicitly.
    if (!rep_) return absl::OkStatus();
    return absl::Status(std::move(*rep_));
  }

  explicit operator bool() const { return !rep_; }

 private:
  // Infrequently set builder options, instantiated lazily. This reduces
  // average construction/destruction time (e.g. the `stream` is fairly
  // expensive). Stacks can also be blown if StatusBuilder grows too large.
  // This is primarily an issue for debug builds, which do not necessarily
  // re-use stack space within a function across the sub-scopes used by
  // status macros.
  struct Rep {
    explicit Rep(absl::Status s) : status(std::move(s)) {}

    explicit operator absl::Status() && {
      std::string message =
          stream.str().empty()
              ? std::string(status.message())
              : absl::StrCat(status.message(), "; ", std::move(stream).str());
      absl::Status ret = absl::Status(status.code(), message);
      if (severity.has_value()) LOG(LEVEL(*severity)) << ret;
      return ret;
    }

    // The status that the result will be based on.  Can be modified by
    // util::AttachPayload().
    absl::Status status;

    std::optional<absl::LogSeverity> severity;

    // Gathers additional messages added with `<<` for use in the final status.
    std::stringstream stream;
  };

  static std::unique_ptr<Rep> InitRep(absl::Status s) {
    if (s.ok()) {
      return nullptr;
    } else {
      return std::make_unique<Rep>(std::move(s));
    }
  }
  std::unique_ptr<Rep> rep_;
};

}  // namespace dxs::status_macros_internal

#define STATUS_MACROS_IMPL_ELSE_BLOCKER_ \
  switch (0)                             \
  case 0:                                \
  default:  // NOLINT

#define RETURN_IF_ERROR(expr)                                                \
  STATUS_MACROS_IMPL_ELSE_BLOCKER_                                           \
  if (auto builder = ::dxs::status_macros_internal::StatusBuilder((expr))) { \
  } else /* NOLINT */                                                        \
    return std::move(builder)

#define STATUS_MACROS_IMPL_CONCAT_INNER_(x, y) x##y
#define STATUS_MACROS_IMPL_CONCAT_(x, y) STATUS_MACROS_IMPL_CONCAT_INNER_(x, y)
#define STATUS_MACROS_IMPL_VARNAME_(var) \
  STATUS_MACROS_IMPL_CONCAT_(var, __LINE__)

#define STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_(statusor, lhs, rexpr, \
                                             error_expression)     \
  auto statusor = (rexpr);                                         \
  if (ABSL_PREDICT_FALSE(!statusor.ok())) return error_expression; \
  lhs = (*std::move(statusor))

#define STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_3_(lhs, rexpr, error_expression) \
  auto STATUS_MACROS_IMPL_VARNAME_(_value_or) = (rexpr);                     \
  if (ABSL_PREDICT_FALSE(!STATUS_MACROS_IMPL_VARNAME_(_value_or).ok())) {    \
    ::dxs::status_macros_internal::StatusBuilder _(                          \
        std::move(STATUS_MACROS_IMPL_VARNAME_(_value_or)).status());         \
    return error_expression;                                                 \
  }                                                                          \
  lhs = (*std::move(STATUS_MACROS_IMPL_VARNAME_(_value_or)))

#define STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_2_(lhs, rexpr) \
  STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_3_(lhs, rexpr, std::move(_))

#define STATUS_MACROS_IMPL_GET_VARIADIC_HELPER_(_1, _2, _3, NAME, ...) NAME
#define STATUS_MACROS_IMPL_GET_VARIADIC_(args) \
  STATUS_MACROS_IMPL_GET_VARIADIC_HELPER_ args

#define ASSIGN_OR_RETURN(...)                                                \
  STATUS_MACROS_IMPL_GET_VARIADIC_((__VA_ARGS__,                             \
                                    STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_3_,  \
                                    STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_2_)) \
  (__VA_ARGS__)

#endif  // DXS_CLIENT_OSS_STATUS_MACROS_H_
