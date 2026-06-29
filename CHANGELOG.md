<!--
SPDX-FileCopyrightText: 2025 EBFM Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# v0.7.0

* Moved performance and profiling sections out of `README.md` into `docs/Performance.md`, together with the GPU setup and CUDA runtime troubleshooting entry. https://github.com/EBFMorg/EBFM/pull/161
* Removed the `uniform` percolation scheme (not used in practice, `normal` and `bucket` are typically chosen). `phys["percolation"]` now accepts `bucket`, `normal` and `linear`. Adapted tests. https://github.com/EBFMorg/EBFM/pull/160
* Fixed `phys["percolation"] = "uniform"` raising `TypeError` on the first timestep in `LOOP_SNOW.py` (only in NumPy path). `test_loop_snow_percolation.py` added. https://github.com/EBFMorg/EBFM/pull/158
* Introduced `--with-gpu` to offload the kernels compaction, heat conduction, percolation to a GPU via `numba.cuda` (NVIDIA) or `numba.hip` (AMD), with `--gpu-vendor {auto,nvidia,amd}` to select/guard the vendor. Mutually exclusive with `--with-numba`. Install via `pip install ebfm[gpu]`. https://github.com/EBFMorg/EBFM/pull/157
* Added `tests/core/test_loop_snow_gpu.py`: runs the snow model on the NumPy and GPU backends from an identical state and requires agreement to `atol=1e-12, rtol=1e-9`. Uses numba's CUDA simulator, so it needs no GPU. https://github.com/EBFMorg/EBFM/pull/157
* Further optimizations in CPU/host-side code in `LOOP_SNOW.py`: fewer full-grid copies, heat conduction precompute moved into Numba kernel. https://github.com/EBFMorg/EBFM/pull/147.
* The `performance` extra no longer installs `mpi4py`: Numba parallelises with threads inside a single process and does not require MPI. Use `EBFM[performance,mpi]` to combine both. https://github.com/EBFMorg/EBFM/pull/147.
* Fixed bug where the shading look-up table for MATLAB grids was pre-computed during initialization even when shading is disabled.  https://github.com/EBFMorg/EBFM/pull/155.
* EBFM now officially support and tests Python 3.14. https://github.com/EBFMorg/EBFM/pull/156
* Fixed bug where resuming from a restart file (`--restart-init`) could lead to time step sizes equal to zero in `LOOP_SNOW.py`. Loading a restart file now asserts the restart variables to contain no missing values and converts them to plain `ndarray`. https://github.com/EBFMorg/EBFM/pull/154
* Fixed bug in icon_to_atmo.py to convert pr_snow units to mwe per timestep (rather than using kg m-2 s-1). https://github.com/EBFMorg/EBFM/pull/149
* Add general interface for definition of fallback values if a coupled component does not provide expected data for individual fields. https://github.com/EBFMorg/EBFM/pull/146.
    * The following fields expected from the component "icon_atmo" will use fallback value if they are missing: "rlds", "clt", "sfcwind", "huss", and "sfcpres"
* Shift coupler init to beginning of execution to reduce wait time for other components. https://github.com/EBFMorg/EBFM/pull/151

# v0.6.1

* Catch invalid values returned by YAC's get and forward info to user. https://github.com/EBFMorg/EBFM/pull/145.

# v0.6.0

* Rename "h" received from Elmer to "surface_elevation". https://github.com/EBFMorg/EBFM/pull/124.
* Assure model time is always in UTC+0, allowing removal of the previously hard-coded (time-zone dependent) parameter dT_UTC. https://github.com/EBFMorg/EBFM/pull/138
* Rename `--netcdf-mesh` to `--netcdf-dem-mesh` and `--netcdf-mesh-unstructured` to `--netcdf-dem-mesh-unstructured`. https://github.com/EBFMorg/EBFM/pull/142
* Use ISO8601 datetime as suffix for restart files created with `--restart-dir` option. https://github.com/EBFMorg/EBFM/pull/141

# v0.5.0

* Add support for ISO8601 format for `--time-step`, `--time-start`, and `--time-end`. This is the recommended format and alternatives are deprecated. https://github.com/EBFMorg/EBFM/pull/137
* Refactor how parser and configuration modules process errors in user input. https://github.com/EBFMorg/EBFM/pull/131

# v0.4.0

* Introduce coarse-resolution (10-km) and fine-resolution (2.5-km) test cases for Greenland forced with meteorological data from CARRA2. https://github.com/EBFMorg/EBFM/pull/128
* Call `def_datetime` and `def_calendar` in YAC coupler setup to forward EBFM calendar and time frame to YAC. This will lead to a YAC error the setup of the coupled run is inconsistent. https://github.com/EBFMorg/EBFM/pull/125.
* Add Elmer Greenland Mesh in this repository for more conveniently running examples. https://github.com/EBFMorg/EBFM/pull/120.
* Coupler `put`/`get` operations now log a warning if coupler returns non-zero error code to investigate unexpected behavior, instead of silently ignoring it. Warning is intentionally non-fatal, because error may be transient or coupler-specific.
* Revise helper script `reader.py`. https://github.com/EBFMorg/EBFM/pull/119.
* Add reduced-size BedMachine Greenland NetCDF example (`examples/BedMachineGreenland-v5_lo.nc`) and two utility scripts under `tools/`: `nc_reduce_size.py` to produce smaller NetCDF copies (field selection and grid subsampling), and `nc_2_vtk.py` to convert NetCDF fields to VTK for visualisation in ParaView. https://github.com/EBFMorg/EBFM/pull/123.
* Bug fixes in double depth method in INIT.py and LOOP_SNOW.py
* Implement MPI handshake for comm splitting. https://github.com/EBFMorg/EBFM/pull/88.
* EBFM now only adds metadata to fields where this is explicitly specified. Note: This can lead to failures in components that do not properly guard `get_metadata` calls with `has_metadata` checks. https://github.com/EBFMorg/EBFM/pull/102
* Add functionality for (optional) unit conversion of data received from/sent to other components. https://github.com/EBFMorg/EBFM/pull/106
* Introduce option `--component-name` to allow configuration of the name this component used to identify to the coupler. https://github.com/EBFMorg/EBFM/pull/101
* Introduced options:
  * `--diagnostics` to show diagnostics for every timestep
  * `--dump-reference` to create file at the end of the run for comparison
  * `--random-seed` to fix the random seed for reproducible results
  * `--with-numba` and `--numba-threads` to run numba kernels with N threads
* Added `tools/compare_snapshots.py` to compare two runs using dumped `.npz` files
* Performance improvements:
  * Improvements in LOOP_SNOW.py (compaction, heat_conduction, percolation_refreezing_and_storage and layer_merging_and_splitting)
  * Added numba kernels for compaction, heat_conduction and percolation_refreezing_and_storage (in `LOOP_SNOW_kernels.py`) , addresses: https://github.com/EBFMorg/EBFM/issues/55
  * Introduced `compute_backend.py` to manage compute-backend dispatch to separate kernel code from logic. Explicit `if/else` dispatch with a single return per function. Prepares codebase for adding, e.g., GPU offload backends without structural changes.

# v0.3.0

* Add `FakeCoupler` for easier testing of coupled workflow. Activated with option `--fake-coupling`. https://github.com/EBFMorg/EBFM/pull/96
* If DoFs are defined with a Elmer mesh locate them at triangle centers (previously: triangle vertices) to allow for conservative mapping schemes that require information about area per DoF. https://github.com/EBFMorg/EBFM/pull/83.
* Update code base to Python 3.10 style for typing and enforce via CI and pre-commit hook. https://github.com/EBFMorg/EBFM/pull/98
* Introduce option `--shading`/`--no-shading` to explicitly overwrite default configuration for meshes. https://github.com/EBFMorg/EBFM/pull/94
* Introduce type checking with mypy for `ebfm.coupling` module. https://github.com/EBFMorg/EBFM/pull/92
* Generalize restart by providing additional options `--restart-dir` and `--restart-init`. https://github.com/EBFMorg/EBFM/pull/90
* Introduce `--field-validation-level` to let user specify how EBFM should treat fields that are defined by EBFM but not provided/accepted by the coupled component. https://github.com/EBFMorg/EBFM/pull/87.
* Fix put/get signatures of couplers and return types to match the Coupler base class,
* Dropped Python 3.9 support in favor of Python >= 3.10 (required for PEP 604 union type annotations). https://github.com/EBFMorg/EBFM/pull/82
* Added tox testing infrastructure with multi-version Python support (3.9-3.13) and separate unit/example test environments. https://github.com/EBFMorg/EBFM/pull/78.
* Introduce `--elmer-mesh-crs-epsg` to let user define the projection used in the Elmer mesh. Mandatory when using `--elmer-mesh`. https://github.com/EBFMorg/EBFM/pull/86.
* Implement a faster shading calculation method based on look-up tables generated before the time-loop. https://github.com/EBFMorg/EBFM/pull/60

# v0.2.0

* Fix and extend `reader.py`, documentation on how to use it and how to obtain required example data. https://github.com/EBFMorg/EBFM/pull/69.
* Revise folder layout to avoid clutter in `site-packages`. Installing EBFM should now only affect `site-packages/ebfm`. https://github.com/EBFMorg/EBFM/pull/73.
* Require Python minimum version 3.9. (Planned to increase to 3.10 soon)
* Clarification how `--start-time` and `--end-time` is interpreted by EBFM. Require that difference of start and end time is a multiple of `--time-step`. https://github.com/EBFMorg/EBFM/pull/58.
* Allow logger configuration via command-line interface. Refer to `ebfm --help` and the options `--log-level-console` and `--log-file`. See https://github.com/EBFMorg/EBFM/pull/56.
* Support new input mesh format. EBFM now accepts Elmer/Ice mesh file for xy-coordinates and separate unstructured NetCDF elevation file obtained from XIOS. To use this feature please provide `--elmer-mesh` together with the new option `--netcdf-mesh-unstructured`. See https://github.com/EBFMorg/EBFM/pull/12.
* Use `setuptools_scm` as backend for `--version` information. See https://github.com/EBFMorg/EBFM/pull/46.
* Remove `pathlib` from requirements, because this can lead to a bug. https://github.com/EBFMorg/EBFM/pull/48.

# v0.1.0

* Initial release
