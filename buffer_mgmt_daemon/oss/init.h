/*
 * Copyright 2025 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 */

#ifndef BUFFER_MGMT_DAEMON_OSS_INIT_H_
#define BUFFER_MGMT_DAEMON_OSS_INIT_H_

namespace tcpdirect {
// Initialize RXDM for open source builds.
void InitRxdm(int argc, char* argv[]);
}  // namespace tcpdirect

#endif  // BUFFER_MGMT_DAEMON_OSS_INIT_H_
