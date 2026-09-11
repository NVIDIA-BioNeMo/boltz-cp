#!/bin/bash
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

# fold-cp SLURM probe — request one node + N GPUs and run the CP environment
# checks there. Fill the <PLACEHOLDERS> with values the user provides, then:
#     sbatch slurm_probe.sbatch
#     squeue -u $USER          # watch
#     cat cp_probe_<jobid>.log # READ the log; do not assume success
#
# mp.spawn fans out the ranks inside a single srun task, so we only need one
# task with all N GPUs visible.

#SBATCH --job-name=fold-cp-probe
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:<N_GPUS>
#SBATCH --cpus-per-task=<CPUS>
#SBATCH --time=00:20:00
#SBATCH --output=cp_probe_%j.log

set -euo pipefail

# --- environment activation (one of these; ask the user which) -------------
# module load cuda/<VERSION>
# source activate <CONDA_ENV>           # e.g. boltz2-pt29
<ENV_ACTIVATION>

WS=<WORLD_SIZE>                          # from probe_env.py recommendation
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== node: $(hostname) ; GPUs: ==="
nvidia-smi -L
nvidia-smi topo -m || true

echo "=== probe_env ==="
python "${SKILL_DIR}/probe_env.py"

echo "=== collectives ==="
export NCCL_DEBUG=WARN
timeout 120 python "${SKILL_DIR}/test_collectives.py" --world-size "${WS}"

echo "=== dtensor smoke ==="
timeout 120 python "${SKILL_DIR}/test_dtensor_smoke.py" --world-size "${WS}"

echo "=== ALL PROBES PASSED ==="
