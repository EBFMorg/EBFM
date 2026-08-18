<!--
SPDX-FileCopyrightText: 2026 EBFM Authors

SPDX-License-Identifier: BSD-3-Clause
-->

# Performance and Profiling Runs

## Installation for Performance Runs

Install EBFM with the performance features:

```sh
pip install -e .[performance]
```

This additionally installs `numba` to run with multiple CPU-threads.

## Running EBFM with performance optimizations

EBFM now includes several performance improvements and supports several options for performance testing and benchmarking:

- Regular (NumPy) path:

  The default run already includes some performance improvements. Run as usual:

  ```sh
  ebfm --matlab-mesh examples/dem_and_mask.mat
  ```

- Numba kernels:

  To enable Numba-accelerated kernels, use the `--with-numba` flag. You can control the number of threads with `--numba-threads N` (replace `N` with the desired thread count):

  ```sh
  ebfm --matlab-mesh examples/dem_and_mask.mat --with-numba --numba-threads 2
  ```

  Note: If you use more than one thread, you must specify `--numba-threads`. In practice, 2 threads have shown the best performance so far, but optimal settings depend on your hardware and problem size. Feel free to experiment.

- GPU offloading:

  Install the GPU dependencies and use the `--with-gpu` flag to offload kernels from `LOOP_SNOW.py` to a GPU:

  ```sh
  pip install -e .[gpu]
  ebfm --matlab-mesh examples/dem_and_mask.mat --with-gpu
  ```

  The same kernels run on NVIDIA (via `numba.cuda`) and on AMD (via `numba.hip`). The vendor is detected automatically. You can use `--gpu-vendor {auto,nvidia,amd}` to select or guard it explicitly. `--with-gpu` and `--with-numba` are mutually exclusive.

  Note: on a cluster you usually have to load a CUDA/ROCm toolkit and point `CUDA_HOME` at it, otherwise Numba cannot compile the kernels even though the GPU itself is visible. See [CUDA runtime not found](#numba-does-not-find-the-cuda-runtime-cudais_available-is-false) in the troubleshooting section.

  Before a long run you can verify the setup with:

  ```sh
  python -c "from numba import cuda; print(cuda.is_available())"
  ```

  At startup `--with-gpu` reports which stack and device it found, so you can check it went to the GPU you expected:

  ```
  [GPU] backend enabled (nvidia). Device: NVIDIA A100-SXM4-80GB  free=79.15 GiB  total=79.15 GiB
  ```

## Timing Your Run

To measure the total runtime, simply prepend your command with `time`:

```sh
time ebfm --matlab-mesh examples/dem_and_mask.mat --with-numba --numba-threads 2
```

## Comparing Results

A script is provided to compare model output snapshots:

- `tools/compare_snapshots.py` can be used to compare two output files (e.g., from different runs or configurations).

To create a reference file for comparison, use the `--dump-reference` option at the end of your run:

```sh
ebfm --matlab-mesh examples/dem_and_mask.mat --dump-reference reference_run.npz
```

Then compare with:

```sh
python tools/compare_snapshots.py reference_run.npz new_run.npz
```

Note: If you use the random-forcing within EBFM in your testcases, make sure to additionally set the option `--random-seed`, as explained below.

## Diagnostics and Reproducibility

- The option `--random-seed` allows reproducible runs (especially if the random forcing in the example testcase is used). Set a fixed seed to ensure identical results for repeated runs (important for benchmarking and debugging):

  ```sh
  ebfm --matlab-mesh examples/dem_and_mask.mat --random-seed 42
  ```

- Use `--diagnostics` to print some diagnostics for a quick overview for every timestep:

  ```sh
  ebfm --matlab-mesh examples/dem_and_mask.mat --diagnostics
  ```

## Troubleshooting

### Numba does not find the CUDA runtime (`cuda.is_available()` is `False`)

*Problem:* `--with-gpu` fails, or `python -c "from numba import cuda; print(cuda.is_available())"` prints `False`, even though the GPU is visible. Note that `cuda.detect()` and `nvidia-smi` can still report the device correctly: they only need the CUDA *driver*, while Numba additionally needs the CUDA *runtime* (`libnvvm`, `libcudart`) to compile kernels.

*Solution:* Point `CUDA_HOME` at a CUDA toolkit installation and add its libraries to `LD_LIBRARY_PATH`. On Levante, for example:

```sh
export CUDA_HOME=/sw/spack-levante/nvhpc-22.5-v4oky3/Linux_x86_64/22.5/cuda/11.7
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Afterwards `cuda.is_available()` should print `True`.
