<!--
SPDX-FileCopyrightText: 2025 EBFM Authors

SPDX-License-Identifier: BSD-3-Clause
-->

# Energy balance and firn model (EBFM)

This repository provides a Python implementation of a energy balance and firn model (EBFM).

Using YAC this model can be coupled to other models.

## Preparations

Please use the script `setup_venv.sh` to create a virtual environment for
developing and running this dummy.

## Running

Activate the `venv` by running `source .venv/bin/activate`.

A basic, uncoupled simulation can be run with the following command:

```sh
python3 src/dummy.py --matlab-mesh examples/dem_and_mask.mat
```

### Mesh data

The arguments `--matlab-mesh`, `--elmer-mesh`, and `--netcdf-mesh` allow to
provide different kinds of mesh data. EBFM supports the following formats:

* MATLAB Mesh: An example is given in `examples/dem_and_mask.mat`. This mesh
  file provides x-y coordinates and elevation data. Please use the argument
  `--matlab-mesh /path/to/your/mesh.mat`.

  Usage example:

  ```sh
  python3 src/dummy.py --matlab-mesh examples/dem_and_mask.mat
  ```

* Elmer Mesh: An Elmer mesh file with x-y coordinates of mesh points and
  elevation data stored in the z-component. Please use the argument
  `--elmer-mesh /path/to/your/elmer/MESH`.

  Usage example:

  ```sh
  python3 src/dummy.py --elmer-mesh examples/DEM
  ```

* Elmer Mesh with Elevation data from NetCDF: The Elmer mesh file provides x-y
  coordinate. An additioal NetCDF file is given to provide elevation data for
  these x-y coordinates. Please use the arguments `--elmer-mesh /path/to/your/elmer/MESH`
  and `--netcdf-mesh /path/to/your/elevation.nc`

  Usage example:

  ```sh
  python3 src/dummy.py --elmer-mesh examples/MESH --netcdf-mesh examples/BedMachineGreenland-v5.nc
  ```

Note that an Elmer mesh must be provided in a directory following the structure:

```
path/to/your/elmer/MESH/
├── mesh.boundary
├── mesh.elements
├── mesh.header
└── mesh.nodes
```

The option `--is-partitioned-elmer-mesh` will tell EBFM that the provided Elmer
mesh is a partitioned mesh. A partitioned mesh file follows the structure:

```
path/to/your/elmer/MESH
    ├── mesh.boundary
    ├── mesh.elements
    ├── mesh.header
    ├── mesh.nodes
    └── partitioning.128
        ├── part.1.boundary
        ├── part.1.elements
        ├── part.1.header
        ├── part.1.nodes
        ...
        ├── part.128.boundary
        ├── part.128.elements
        ├── part.128.header
        └── part.128.nodes
```

Usage example for partitioned mesh:

```sh
python3 src/dummy.py --elmer-mesh examples/MESH/partitioning.128/ --netcdf-mesh examples/BedMachineGreenland-v5.nc --is-partitioned-elmer-mesh --use-part 42
```

### Coupled simulation

The EBFM code allows coupling to other simulation codes. The following arguments
allow to configure the coupling:

```sh
python3 src/dummy.py ...
  --couple-to-elmer-ice
  --couple-to-icon-atmo
  --coupler-config /path/to/coupling/config.yaml
```

Note that the coupling uses the Python bindings of YAC. Additionally, EBFM must
be run in a MPMD (multiple process multiple data) run. An example command for
running a coupled simulation with Elmer/Ice and ICON is given below:

```sh
mpirun -n 1 python $EBFM_ROOT/src/dummy.py \
  --elmer-mesh $MESHES/MESH/partitioning.128/ \
  --netcdf-mesh $DATA/BedMachineGreenland-v5.nc \
  --is-partitioned-elmer-mesh --use-part 1 \
  --coupler-config $CPL_CONFIG --couple-to-elmer --couple-to-icon \
  : \
  -n 1 $ELMER_ROOT/src/elmer_dummy_f.x $MESHES/MESH/partitioning.128 1 $CPL_CONFIG \
  : \
  -n 1 $ICON_ROOT/src/icon_dummy.x $MESHES/icon_grid_0054_R02B08_G.nc $DATA/mbe3064_atm_elmer_monmean_1979.nc $DATA/varlist_elmerfile
```

Be aware that the command above requires setting a few environment variables.
Assuming your project is structured following [this repository](https://gitlab.dkrz.de/k202215/ebfm_dummy)
the following settings should help getting started:

```sh
export EBFM_DUMMY_REPO=/path/to/k202215/ebfm_dummy
export EBFM_REPO=/path/to/this/repo

export CPL_CONFIG=$EBFM_DUMMY_REPO/config/coupling.yaml
export MESHES=$EBFM_DUMMY_REPO/mesh
export DATA=$EBFM_DUMMY_REPO/data

export EBFM_ROOT=$EBFM_REPO
export ELMER_ROOT=$EBFM_DUMMY_REPO/dummies/ELMER
export ICON_ROOT=$EBFM_DUMMY_REPO/dummies/ICON
```

Depending on the binaries that you want to use `$ELMER_ROOT` and/or `$ICON_ROOT`
may be set to point to the non-dummy versions of the codes.

## Troubleshooting

### `libnetcdf.so` not found at runtime

*Problem:* `./icon_dummy.x: error while loading shared libraries: libnetcdf.so.15: cannot open shared object file: No such file or directory`
*Solution:* Try rebuilding `icon_dummy.x`
`
### `libyaxt_c.so` not found at runtime

*Problem:* `./icon_dummy.x: error while loading shared libraries: libyaxt_c.so.1: cannot open shared object file: No such file or directory`

*Solution:* `export LD_LIBRARY_PATH='$YAXT_INSTALL_DIR/lib/'

### `#include <proj.h>` not found when building the Elmer dummy

*Problem:*
```sh
...
elmer_grid.c:11:10: fatal error: proj.h: No such file or directory
   11 | #include <proj.h>
      |          ^~~~
compilation terminated.
make: *** [Makefile:48: elmer_grid.o] Error 1
```

*Solution:* `sudo apt-get install libproj-dev`

# Developer notes

## pre-commit

This project uses pre-commit hooks for some tasks described in detail below. To setup pre-commit please do the following:

```sh
pipx install pre-commit
pre-commit install
```

**Alternative:** If you are already working in an virtual environment, you can also use `pip install pre-commit` instead of `pipx`. Refer to the [documentation of `pipx`](https://pipx.pypa.io/stable/#overview-what-is-pipx) for further information.

As soon as pre-commit is set up, you will not be able to commit if any of the checks fails. With the help of the logging output it should usually be possible to fix the problem.

Note: You can bypass this check with `--no-verify`. Please note that the CI will also run pre-commit and fail if there are problems in any of the checks. Therefore, it is recommended to use the pre-commit hooks locally before pushing code to this repository and only bypass them if there is a good reason.

## Copyright and licensing

This project uses [REUSE](https://reuse.software/) to track information regarding copyright and licensing. Therefore, all files in this repository are required to provide the corresponding information. Please refer to the documentation of REUSE for details.

You can use pre-commit to automatically check if all files in the repository provide the necessary information:

```
pre-commit run reuse --all-files
```

## Code formatting

Automated checks for PEP8 compiance are implemented following [^1] with some modifications. You can use pre-commit hooks to automatically format your code with black:

```sh
pre-commit run black --all-files
```

With flake8 you can check whether your code follows all relevant formatting rules:

```sh
pre-commit run flake8 --all-files
```

Please note that black might not be able to automatically fix all problems and therefore flake8 might fail even if you have run black before. In this case, you will have to manually fix the remaining problems.

----

[^1]: https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/
