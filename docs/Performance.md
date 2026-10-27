<!--
SPDX-FileCopyrightText: 2026 EBFM Authors

SPDX-License-Identifier: BSD-3-Clause
-->

# Performance and Profiling Runs

## Installation for Performance Runs

Install EBFM with the performance features:

```sh
pip install ebfm[performance]
```

This additionally installs `numba` to run with multiple CPU-threads.

For GPU offloading, install the GPU dependencies instead:

```sh
pip install ebfm[gpu]
```

Note: on a cluster you usually have to load a CUDA/ROCm toolkit and point `CUDA_HOME` at it, otherwise Numba
cannot compile the kernels even though the GPU itself is visible. See
[CUDA runtime not found](#numba-does-not-find-the-cuda-runtime-cudais_available-is-false) below.

It is recommended to verify your cuda setup by running:

```sh
python -c "from numba import cuda; print(cuda.is_available())"
# expected output: True
```

## Running EBFM with performance optimizations

EBFM offers several options for tuning performance on a given system and for testing and benchmarking it.

### Numba kernels: `--with-numba`

The option `--with-numba` runs the `LOOP_SNOW.py` kernels (compaction, heat conduction, percolation) as compiled,
multi-threaded Numba kernels instead of NumPy array operations. Use it on a multi-core CPU when no GPU is available.
It generally outperforms the NumPy path, the more clearly the larger the mesh and the longer the run.

It is not on by default because `numba` is an optional dependency and the kernels compile on first use. A very short run can therefore spend more time compiling than it saves.

Control the number of threads with `--numba-threads N` (replace `N` with the desired thread count).

**Example:** Run EBFM on the MATLAB example mesh with the Numba kernels on two threads:

```sh
ebfm --matlab-mesh examples/dem_and_mask.mat --with-numba --numba-threads 2
```

Note: If you use more than one thread, you must specify `--numba-threads`. In practice, 2 threads have shown the
best performance so far, but optimal settings depend on your hardware and problem size. Feel free to experiment.

`--numba-threads` is per MPI rank and cannot exceed the number of CPUs the rank may run on. Under MPI and Slurm this
is usually less than the number of CPUs of the node: `mpirun` or `srun` may bind each rank to a few cores, and Slurm
only grants the CPUs of the allocation. EBFM logs Numba's limit and the usable CPUs at startup, and stops with an error
if `--numba-threads` exceeds them (see [Troubleshooting](#troubleshooting)).

### GPU offloading: `--with-gpu`

You can use the option `--with-gpu` when the node you run on has a GPU: it offloads the same kernels to the device as for the option `--with-numba`. How much that
gains depends on the mesh size and the GPU, and for small meshes it is not necessarily faster than `--with-numba`.
The options `--with-gpu` and `--with-numba` tell EBFM to select alternative backends for those kernels and are therefore mutually exclusive.

**Example:** Run EBFM on the MATLAB example mesh with the GPU kernels:

```sh
ebfm --matlab-mesh examples/dem_and_mask.mat --with-gpu
```

The same kernels run on NVIDIA and AMD GPUs, see `--gpu-vendor` in `ebfm --help`.

When `--with-gpu` is set, EBFM will report at startup which vendor backend (`nvidia` or `amd`) and which device have been found, so you can check it went to the GPU you expected:

```
[GPU] backend enabled (nvidia). Device: NVIDIA A100-SXM4-80GB  free=78.83 GiB  total=79.15 GiB
```

## Timing Your Run

To measure the total runtime prepend your command with
[`time`](https://www.gnu.org/software/bash/manual/bash.html#Pipelines):

```sh
time ebfm --matlab-mesh examples/dem_and_mask.mat --with-numba --numba-threads 2
```

The shell then prints the wall-clock (`real`) and CPU times, for example:

```
real    1m12.771s
user    2m18.492s
sys     0m2.113s
```

For benchmarking, compare `real` between runs: `user` exceeding it only means several threads ran in parallel.

## Comparing Results

The script `tools/compare_snapshots.py` allows to compare two output files (e.g., from different runs or configurations).

To create a reference file for comparison, use the `--dump-reference` option at the end of your run:

```sh
ebfm --matlab-mesh examples/dem_and_mask.mat --dump-reference reference_run.npz
```

Then compare with:

```sh
python tools/compare_snapshots.py reference_run.npz new_run.npz
```

Note: uncoupled runs (such as the example testcase) generate their climate forcing randomly, so make sure to additionally set the option `--random-seed`, as explained below.

## Diagnostics and Reproducibility

- The option `--random-seed` allows reproducible runs (especially if the random forcing in the example testcase is used). Set a fixed seed to ensure identical results for repeated runs (important for benchmarking and debugging):

  ```sh
  ebfm --matlab-mesh examples/dem_and_mask.mat --random-seed 42
  ```

- Use `--print-diagnostics` to print some diagnostics for a quick overview for every timestep:

  ```sh
  ebfm --matlab-mesh examples/dem_and_mask.mat --print-diagnostics
  ```

## Troubleshooting

### Numba does not find the CUDA runtime (`cuda.is_available()` is `False`)

*Problem:* `--with-gpu` fails, or `python -c "from numba import cuda; print(cuda.is_available())"` prints `False`, even though the GPU is visible. Note that `cuda.detect()` and `nvidia-smi` can still report the device correctly: they only need the CUDA *driver*, while Numba additionally needs the CUDA *runtime* (`libnvvm`, `libcudart`) to compile kernels.

*Solution:* Point `CUDA_HOME` at a CUDA toolkit installation and add its libraries to `LD_LIBRARY_PATH`. On Levante, for example:

```sh
export CUDA_HOME=/sw/spack-levante/nvhpc-22.5-v4oky3/Linux_x86_64/22.5/cuda/11.7
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Afterwards the check above should succeed:

```sh
python -c "from numba import cuda; print(cuda.is_available())"
# expected output: True
```

### `--numba-threads=N exceeds Numba's thread limit`

*Problem:* EBFM stops at startup with

```
error: --numba-threads=8 exceeds Numba's thread limit of 2: this process may only run on 2 CPUs. ...
```

Numba can run at most as many threads as the CPUs the process is allowed to run on (its CPU affinity), or as many as
the environment variable `NUMBA_NUM_THREADS` allows if it is set. Under MPI the affinity is often much smaller than the
node: OpenMPI binds each rank to a single core by default when it starts 2 or fewer processes, e.g. one EBFM rank plus
one coupled component. With `srun`, each task may only run on the CPUs it requested with `--cpus-per-task`. On Levante
a core has 2 hardware threads and Slurm allocates whole cores, hence a limit of 2 for a single core or
`--cpus-per-task=1`.

To check the limit Numba sees, launch Python exactly like EBFM, e.g. with `mpirun` or `srun`:

```sh
mpirun -n 1 python -c "import os, numba; print(len(os.sched_getaffinity(0)), numba.config.NUMBA_NUM_THREADS)"
srun -n 1 python -c "import os, numba; print(len(os.sched_getaffinity(0)), numba.config.NUMBA_NUM_THREADS)"
```

*Solution:* Use one of

- `mpirun --bind-to none ...`: each rank may run on all CPUs of the Slurm allocation. Simple, and suitable for
  interactive runs with few ranks per node; ranks on the same node then share these CPUs.
- `mpirun --map-by slot:PE=N ...` with an allocation that has enough CPUs per task (e.g. `salloc -c ...`): each rank
  is bound to its own N cores. Preferable for production runs with many ranks per node.
- with `srun`: `--cpus-per-task=N` for the job and for `srun` itself, e.g.
  `srun --cpus-per-task="$SLURM_CPUS_PER_TASK" ...` in a batch script. On Levante N counts hardware threads.
- a lower `--numba-threads`.

If the error names `NUMBA_NUM_THREADS`, that environment variable is set below `--numba-threads`: raise or unset it.
