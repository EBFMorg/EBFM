# EBFM dummy

This dummy provides a template how EBFM can be coupled to other models via YAC.

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

## Code formatting

Automated checks for PEP8 compiance are implemented following [^1] with some modifications. Please install pre-commit first:

```sh
pipx install pre-commit
```

Then, install the pre-commit hooks by running the following command from this folder:

```sh
pre-commit install
```

The pre-commit hooks will now run automatically on every commit. 

Note: You can bypass this check with `--no-verify`. Please note that the CI will also check for PEP8 compliance and fail if the code is not PEP8 compliant. Therefore, it is recommended to use the pre-commit hooks locally before pushing code to this repository.

----

[^1]: https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/
