<!--
SPDX-FileCopyrightText: 2025 EBFM Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# develop

* Add a 10-km test case for Greenland forced with CARRA2 meteorological data for May 1990 (`examples/Greenland_CARRA2`). https://github.com/EBFMorg/EBFM/pull/128
    * Add feature to use NetCDF meshes with DEM as input (`--netcdf-mesh`)
    * Add option to set missing elevations of NetCDF meshes to 0 m (`--fill-missing-elevation`)
    * Add feature to read forcing files from folder (`--forcing-dir`)
* Remove support for Python 3.10. https://github.com/EBFMorg/EBFM/pull/143
* `--numba-threads` is now checked against the number of threads Numba accepts, i.e. the CPUs the rank may run on (CPU affinity from MPI binding or the Slurm allocation, e.g. `--cpus-per-task`) or the `NUMBA_NUM_THREADS` environment variable. Exceeding it stops EBFM with an error explaining the cause, instead of a Numba `ValueError` traceback. Exceeding the suggested thread count for the ranks sharing a node is now only a warning; the suggestion no longer divides the CPUs of the whole node by the number of EBFM ranks. https://github.com/EBFMorg/EBFM/pull/190
* A fatal error now ends EBFM with exit status 1 whenever mpi4py is installed, both standalone and under `mpirun`. EBFM previously aborted MPI with error code 0, so the run, and `mpirun` with it, reported success after logging "Fatal Error in EBFM". The same applies to an MPI handshake version mismatch. A fatal error before the MPI communicator splitting no longer leaves the other components of an MPMD run waiting forever. https://github.com/EBFMorg/EBFM/pull/191
* Add an example for `--netcdf-dem-mesh-unstructured`: surface elevation and ice thickness from Elmer/Ice XIOS output for the Greenland mesh (`examples/greenland_mesh_v0/xios_dem.nc`). `examples/README.md` describes the contents EBFM expects in such a file. https://github.com/EBFMorg/EBFM/pull/198
* Add safeguards for illegal MPI runs. https://github.com/EBFMorg/EBFM/pull/201
    * EBFM on several MPI ranks now stops at startup with an error unless it runs on a partitioned Elmer mesh (`--is-partitioned-elmer-mesh`), one partition per rank. With `--matlab-mesh`, `--netcdf-mesh`, or `--elmer-mesh` without `--is-partitioned-elmer-mesh`, every rank previously computed the whole grid, and the run failed later, e.g. while the ranks wrote the same output file.
    * EBFM on several MPI ranks now stops at startup with an error if `--use-part` is given. Every rank previously computed the same partition.
    * EBFM on several MPI ranks now stops at startup with an error if forcing files (`--forcing-dir`) or restart files (`--restart-dir`, `--restart-init`) are used, also on a partitioned Elmer mesh. On a partitioned mesh, `--forcing-dir` previously failed when reading the first forcing file, `--restart-init` failed at the shape check of the restart file, and `--restart-dir` finished without writing a restart file, with a warning at the end. EBFM on a single rank, e.g. as one component of a coupled MPMD run, is unaffected.
    * EBFM started by an MPI launcher on several processes (e.g. `mpirun -np 4 ebfm ...`) without mpi4py now stops with an error asking to install `ebfm[mpi]`. Every process previously ran the whole model on its own, and the processes then failed while writing the same output file. Detected via the launcher's `OMPI_COMM_WORLD_SIZE` (Open MPI) or `PMI_SIZE` (MPICH, Intel MPI, `srun --mpi=pmi2`).
* The results of a run on a partitioned Elmer mesh (`--is-partitioned-elmer-mesh`) on several MPI ranks no longer depend on the partitioning. https://github.com/EBFMorg/EBFM/pull/203
    * The mean latitude that selects the water vapour transmissivity of the incoming shortwave radiation is now computed over the whole domain. Each rank previously used the mean latitude of its own partition.
    * The lapse rate of potential temperature, used by the sensible and latent heat fluxes, is now fitted over the whole domain. Each rank previously fitted it over its own partition.
    * Without `--random-seed`, rank 0 now draws the random seed and shares it with all ranks, so the random forcing is the same over the whole domain. Each rank previously drew its own.
    * `--dump-reference` now holds the columns of all ranks, ordered by Elmer element id. Every rank previously wrote its own partition to the same file.
    * `--print-diagnostics` now reports the column count and `smb` statistics of the whole domain, logged by rank 0. Each rank previously reported its own partition.
* In a run on several MPI ranks, only rank 0 logs to the console at `--log-level-console`; the other ranks only log warnings and errors there, instead of repeating every message. Log files (`--log-file`, one per rank) are unchanged. The current model time is logged instead of printed. https://github.com/EBFMorg/EBFM/pull/204
* A MATLAB grid (`--matlab-mesh`) can now run on several MPI ranks, e.g. `mpirun -np 4 ebfm --matlab-mesh examples/dem_and_mask.mat --no-shading`: each rank computes a contiguous block of the columns, and rank 0 writes the output file, which is identical to that of a run on a single rank. Shading is not supported on several ranks and stops the run with an error. https://github.com/EBFMorg/EBFM/pull/202
* A usage error, e.g. a rejected command line option, now ends the whole MPI job with its exit status when `MPI.COMM_WORLD` has more than one process. EBFM previously exited on its own, and the other processes, e.g. a coupled model in an MPMD run, waited for it forever. A run on a single process still exits without `MPI_ABORT`. https://github.com/EBFMorg/EBFM/pull/206
* Fixed the slope azimuth of grid cells on MATLAB and NetCDF meshes that face exactly east or west (north-south gradient exactly zero): they were treated as facing the opposite direction, so the incoming shortwave radiation of an east-facing cell peaked in the afternoon instead of the morning. Results change in these cells. https://github.com/EBFMorg/EBFM/pull/194
* Grids built from `--elmer-mesh` without a DEM now place each column at the latitude and longitude of its mesh cell center, as they already did with `--netcdf-dem-mesh`. Every column previously sat at 75 N 40 W regardless of the mesh. Results of such runs change through the solar geometry, unless the surface energy balance is taken from ICON-Land (`--couple-to-icon-land`). The `lon` written to the output now ranges from -180 to 180 degrees east instead of being 320. https://github.com/EBFMorg/EBFM/pull/193
* Grids built from `--elmer-mesh` alone (no `--netcdf-dem-mesh`) now take each column's elevation from the mesh: the mean of the z coordinates of its cell's vertices, as `reader.py` writes them from a DEM. The elevation was previously random between 0 and 100 m. A mesh without elevation data, such as `examples/greenland_mesh_v0/MESH`, now gives 0 m everywhere. https://github.com/EBFMorg/EBFM/pull/193
* Grids built from `--elmer-mesh` alone (no `--netcdf-dem-mesh`) now take each column's surface slope and orientation from the plane through the vertices of its mesh cell, instead of treating every column as flat. This changes the shortwave radiation on sloping columns. A mesh without elevation data still gives flat columns. When coupled to Elmer/Ice (`--couple-to-elmer-ice`), the columns stay flat and a warning is logged, since the slopes do not follow the surface elevation received from Elmer/Ice yet. https://github.com/EBFMorg/EBFM/pull/193
* In a run on several MPI ranks, EBFM now checks before the time loop that every column is held by exactly one rank. The check communicates the ids of all columns; `--no-check-grid-distribution` skips it. https://github.com/EBFMorg/EBFM/pull/XXX
* Partitioned Elmer meshes (`--is-partitioned-elmer-mesh`) now write model output (`Output/model_output.nc`); it was skipped before. On several MPI ranks, rank 0 gathers the cells of all ranks and writes a single file, with the cells ordered by Elmer element id and the vertices that partitions share written once. https://github.com/EBFMorg/EBFM/pull/XXX

# v0.8.0

* A `put()` or `get()` for a field name that the coupled component never registered now logs a warning listing the registered field names and skips the operation, returning the new exit code `UNREGISTERED_FIELD`. With `--fake-coupling` such a call was previously discarded (`put`) or answered with a no-data result indistinguishable from a field that is simply not coupled (`get`); with YAC it failed with a `KeyError`. Skipping instead of failing keeps a field that is deliberately left undefined harmless, e.g. Elmer/Ice's `dhdx`/`dhdy`. https://github.com/EBFMorg/EBFM/pull/185
* `--fake-coupling` now works for grids built from an Elmer mesh, instead of failing during coupler setup with "Could not infer number of grid points". The number of points is taken from the per-column fields of the grid, which is what the coupled fields are sized by. https://github.com/EBFMorg/EBFM/pull/184
* The grid elevation now follows the surface elevation received from Elmer/Ice in every Elmer-coupled run. It previously only did so when coupled to ICON as well, and then assigned a single column's elevation to the whole grid, which pinned the potential temperature lapse rate to its fallback value and wrote a constant elevation to the output file. Results of Elmer-coupled runs change accordingly. Quantities derived from the elevation during initialization do not follow yet, so a run now stops with an `AssertionError` when one of them is in use: shading, non-zero surface slopes (Elmer/Ice does not send `dhdx`/`dhdy` yet), or a 2-D elevation field as MATLAB grids have. https://github.com/EBFMorg/EBFM/pull/182
* The exchange with a coupled component can now be split into multiple phases (e.g. `send -> do computation -> receive`). `Component.exchange` accepts an optional `get_keys` argument listing the fields to be received, next to `data_to_exchange` which defines the fields to be sent. A component may internally enforce additional conditions (i.e., collections of send/receive pairs that must be communicated together) and raise a `ValueError` if requirements are not fulfilled. `icon_atmo` and `elmer_ice` support expect a single exchange where all fields are exchanged in one phase, as before. https://github.com/EBFMorg/EBFM/pull/175
* NetCDF output writing now supports unstructured Elmer-based grids in addition to MATLAB grids (previously skipped with a warning), including mesh topology when available. Output is now written for any non-partitioned run regardless of grid type or coupling, with a one-time startup warning that non-structured or coupled output is not yet covered by tests. https://github.com/EBFMorg/EBFM/pull/135
* Added coupling to the ICON land model (ICON-Land/JSBACH) as a separate component (`--couple-to-icon-land`, component `icon_land`): EBFM sends the ice-covered fraction of its grid cells (`icefract`) and the surface albedo (`albedo`). The provisional albedo field of the `icon_atmo` component was removed. The abbreviation `--couple-to-icon` is now ambiguous and must be spelled out as `--couple-to-icon-atmo`. https://github.com/EBFMorg/EBFM/pull/178
* The `icon_land` component also receives the surface energy balance results of ICON-Land's glacier tile, averaged over the EBFM time step: `t_srf` (K), `melt` and `evapotrans` (converted to m w.e. per time step, evapotrans negative upward). When received, they drive the snow/firn model and the mass balance instead of EBFM's own energy balance (`IconLand.map_energy_balance_to_ebfm`): `Tsurf`, `melt` and the `moist_*` terms are taken from ICON-Land (evapotranspiration partitioned by the surface temperature as before). EBFM's own energy balance is still computed as a diagnostic and kept as `ebm_Tsurf`, `ebm_melt`, `ebm_Emelt` (written to the output) and `ebm_moist_*`. Not receiving one of the three fields aborts the run, since the firn model cannot be driven by a partial surface energy balance; `--fake-coupling` provides them as constants (260 K, no melt, no evapotranspiration), so that a pseudo-coupled run exercises this path. https://github.com/EBFMorg/EBFM/pull/178
* The `icon_land` component sends EBFM's firn state and mass balance to ICON-Land: `t_sub` (first subsurface layer temperature), `ghf_cond` (conductance, factored out of `LOOP_EBM` into `LOOP_EBM_GHF.conductance`), `runoff`, `smb` (kg m-2 s-1) and `snowmass` (kg m-2). https://github.com/EBFMorg/EBFM/pull/178
* The `icon_land` component now uses the multi-phase exchange above to send its surface/firn state and receive ICON-Land's surface energy balance results as two separate exchanges instead of one: EBFM sends its state, then exchanges with `icon_atmo`, and only then receives the results from `icon_land`, once ICON-Land has had time to compute them. Combining both into a single exchange, as before, is now rejected. The coupling sequence in `main.py` changes accordingly; the exchanged fields and their units are unchanged. https://github.com/EBFMorg/EBFM/pull/178

# v0.7.0

* Loading a restart file (`--restart-init`) now also rejects a file that is missing a required variable, or that stores a per-layer variable per-column (or vice versa). Both previously loaded without complaint and only surfaced much later, if at all. The number of columns and layers a run uses is now logged during initialization. https://github.com/EBFMorg/EBFM/pull/169
* Loading a restart file (`--restart-init`) now rejects files whose arrays do not match the configured column discretization, instead of loading them and mixing two different column or layer counts into the model state. https://github.com/EBFMorg/EBFM/pull/167
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
