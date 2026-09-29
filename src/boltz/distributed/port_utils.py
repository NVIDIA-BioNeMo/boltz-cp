# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Port discovery utilities for distributed init.

Concurrent test worktrees collide on a fixed MASTER_PORT. The canonical mitigation
is to let the OS assign a free port via a bind-to-zero probe, then propagate the
allocated port to all ranks via the MASTER_PORT env var before init_process_group.

The returned port is not reserved after this function closes its socket. Callers
should therefore allocate it as close as possible to process-group initialization.
"""

import socket


def find_free_port() -> int:
    """Return an OS-assigned TCP port available on every local IPv4 address."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # TCPStore rank 0 listens on any local address. Probe the same wildcard
        # namespace rather than loopback alone, which can report a port as free
        # while another local interface already owns it.
        sock.bind(("0.0.0.0", 0))
        return int(sock.getsockname()[1])
