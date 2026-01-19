<!--
SPDX-FileCopyrightText: 2025 EBFM Authors

SPDX-License-Identifier: BSD-3-Clause
-->

# Energy balance and firn model (EBFM)

[![PyPI version](https://img.shields.io/pypi/v/ebfm)](https://pypi.org/p/ebfm/) ![License](https://img.shields.io/pypi/l/ebfm)

This repository provides a Python implementation of an energy balance and firn model (EBFM).

Using YAC, this model can be coupled to other models.

## Basic installation

### Create virtual environment

You can use an arbitrary location to create your virtual environment.
A common choice is `.venv`. In the following, we will use `$VENV` as a placeholder and the path to the virtual environment `.venv` should be set as follows:

```sh
export VENV=/path/to/your/environment/.venv
```

If you do not already have a virtual environment, you can create one using the following commands, which will additionally install the necessary dependencies specified in `pyproject.toml`:

```sh
python3 -m venv $VENV
source $VENV/bin/activate
```

Check if your virtual environment is activated. You should see the name within parentheses just before your command prompt, in this case it should say `(.venv)`.

If you want to know more about virtual environments, please refer to the [Python documentation](https://docs.python.org/3/library/venv.html). \
If you intend to develop EBFM, please take a look at the [developer notes](https://github.com/wardvp/EBFM/edit/matthias-ibach-patch-1/README.md?pr=%2Fwardvp%2FEBFM%2Fpull%2F23#developer-notes) further down in the `README`.

### Install EBFM

You can then install EBFM directly into you virtual environment by running the following command:

```sh
pip3 install EBFM
```

This will install the basic version of EBFM without coupling.

### Checking your installation

Please check your installation by running `ebfm --help` to print the help message and `ebfm --version` to print the installed version.

## Installation with coupling features

Before installing EBFM with coupling features, you must install YAC with the Python bindings. Similar to the instructions above, we will use a virtual environment to install all required dependencies.
During YAC’s configuration step, set the installation prefix to your virtual environment and enable Python bindings (`--prefix=$VENV` `--enable-python-bindings`).
Make sure to do this *after* creating your virtual environment.

The procedure should look similar to below:

```sh
python3 -m venv $VENV
# configure and install YAC with --prefix=$VENV and --enable-python-bindings
source $VENV/bin/activate
pip3 install EBFM[cpl]
```

Adding `[cpl]` will make sure that additional dependencies needed for coupling, such as `yac`, are present in your virtual environment.

If you see any errors during the process, please make sure that your virtual environment is activated. If this is the case, you should see the name within parentheses just before your command prompt, in this case it should say `(.venv)`.

If during the installation of EBFM it appears that `yac` is missing, please double-check with `pip3 freeze` that the package has been installed properly. For detailed instructions on how to install YAC properly and guidance for troubleshooting, see the YAC documentation on [Python bindings](https://yac.gitlab-pages.dkrz.de/YAC-dev/d7/d9e/pythonbindings.html).

## Running EBFM

After installation, a basic, uncoupled simulation can be run with the following command, provided you cloned this repository:

```sh
ebfm --matlab-mesh examples/dem_and_mask.mat
```

### Mesh data

The arguments `--matlab-mesh`, `--elmer-mesh`, and `--netcdf-mesh` allow to provide different kinds of mesh data.
EBFM supports the following formats:

* MATLAB Mesh: An example is given in `examples/dem_and_mask.mat`. This mesh
  file provides x-y coordinates and elevation data. Please use the argument
  `--matlab-mesh /path/to/your/mesh.mat`.

  Usage example:

  ```sh
  ebfm --matlab-mesh examples/dem_and_mask.mat
  ```

* Elmer Mesh: An Elmer mesh file with x-y coordinates of mesh points and
  elevation data stored in the z-component. Please use the argument
  `--elmer-mesh /path/to/your/elmer/MESH`.

  Usage example:

  ```sh
  ebfm --elmer-mesh examples/DEM
  ```

* Elmer Mesh with Elevation data from NetCDF: The Elmer mesh file provides x-y
  coordinate. An additioal NetCDF file is given to provide elevation data for
  these x-y coordinates. Please use the arguments `--elmer-mesh /path/to/your/elmer/MESH`
  and `--netcdf-mesh /path/to/your/elevation.nc`

  Usage example:

  ```sh
  ebfm --elmer-mesh examples/MESH --netcdf-mesh examples/BedMachineGreenland-v5.nc
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
ebfm --elmer-mesh examples/MESH/partitioning.128/ --netcdf-mesh examples/BedMachineGreenland-v5.nc --is-partitioned-elmer-mesh --use-part 42
```

### Using `reader.py` to prepare an Elmer mesh with elevation data

The helper script `reader.py` can be used to combine an existing Elmer mesh with a DEM NetCDF file by replacing the z‑coordinates in `mesh.nodes` ahead of time. This can be useful if you want to preprocess a mesh once and reuse it for multiple EBFM runs.

For example, you can run the following command from the repository root (`src`):

```sh
python3 reader.py ../examples/MESH ../examples/BedMachineGreenland-v5.nc
```

By default, you must choose whether to write the updated mesh to a new directory, where `../examples/MESH_with_DEM` is the path to the new directory (copies your original `MESH` folder):

```sh
python3 reader.py ../examples/MESH ../examples/BedMachineGreenland-v5.nc --outpath ../examples/MESH_with_DEM
```

or modify the mesh in place (overwrites `mesh.nodes`):

```sh
python3 reader.py ../examples/MESH ../examples/BedMachineGreenland-v5.nc --in-place
```

The resulting mesh can then be used directly with EBFM:

```sh
ebfm --elmer-mesh ../examples/MESH_with_DEM
```



### Coupled simulation

The EBFM code allows coupling to other simulation codes. The following arguments
allow to configure the coupling:

```sh
ebfm ...
  --couple-to-elmer-ice
  --couple-to-icon-atmo
  --coupler-config /path/to/coupling/config.yaml
```

Note that the coupling uses the Python bindings of YAC. Additionally, EBFM must
be run in a MPMD (multiple process multiple data) run.
Follow the install instructions from above and run the example command for a coupled simulation with Elmer/Ice and ICON:

```sh
mpirun -np 1 ebfm \
  --elmer-mesh $MESHES/MESH/partitioning.128/ \
  --netcdf-mesh $DATA/BedMachineGreenland-v5.nc \
  --is-partitioned-elmer-mesh --use-part 1 \
  --coupler-config $CPL_CONFIG --couple-to-elmer --couple-to-icon \
  : \
  -np 1 $ELMER_ROOT/src/elmer_dummy_f.x $MESHES/MESH/partitioning.128 1 $CPL_CONFIG \
  : \
  -np 1 $ICON_ROOT/src/icon_dummy.x $MESHES/icon_grid_0054_R02B08_G.nc $DATA/mbe3064_atm_elmer_monmean_1979.nc $DATA/varlist_elmerfile
```

Be aware that the command above requires setting a few environment variables.
Assuming your project is structured following [this repository](https://gitlab.dkrz.de/TerraDT/ebfm_dummy),
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

## Developer notes

### Optional dependencies `[dev]`

Please install the package with the optional dependencies `[dev]` if you are developing (this will automatically
install `pre-commit`).

To concatenate multiple optional dependencies, please run

```sh
pip3 install EBFM[dev,cpl]
```

### pre-commit

This project uses pre-commit hooks for some tasks described in detail below. To setup pre-commit please do the following:

```sh
pipx install pre-commit
pre-commit install
```

**Alternative:** If you are already working in an virtual environment, you can also use `pip3 install pre-commit` instead of `pipx`. Refer to the [documentation of `pipx`](https://pipx.pypa.io/stable/#overview-what-is-pipx) for further information.

As soon as pre-commit is set up, you will not be able to commit if any of the checks fails. With the help of the logging output it should usually be possible to fix the problem.

Note: You can bypass this check with `--no-verify`. Please note that the CI will also run pre-commit and fail if there are problems in any of the checks. Therefore, it is recommended to use the pre-commit hooks locally before pushing code to this repository and only bypass them if there is a good reason.

**Troubleshooting:** The pre-commit hooks require Python >= 3.10. If your Python version is older you will see an error similar to the following

```sh
ERROR: Package 'black' required a different Python: 3.9.9 not in `>=3.10`
```

Please update your Python in this case. Alternatively, you can also set up an independent virtual environment just for running the pre-commit hooks or skip the checks with `--no-verify`.

### Copyright and licensing

This project uses [REUSE](https://reuse.software/) to track information regarding copyright and licensing. Therefore, all files in this repository are required to provide the corresponding information. Please refer to the documentation of REUSE for details.

You can use pre-commit to automatically check if all files in the repository provide the necessary information:

```
pre-commit run reuse --all-files
```

### Code formatting

Automated checks for PEP8 compiance are implemented following [^1] with some modifications. You can use pre-commit hooks to automatically format your code with black:

```sh
pre-commit run black --all-files
```

With flake8 you can check whether your code follows all relevant formatting rules:

```sh
pre-commit run flake8 --all-files
```

Please note that black might not be able to automatically fix all problems and therefore flake8 might fail even if you have run black before. In this case, you will have to manually fix the remaining problems.

### Further hints

* Please consider installing EBFM via `pip3 --editable .` if you are developing the package.

----

[^1]: https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/
