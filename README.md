# EBFM dummy

This dummy provides a template how EBFM can be coupled to other models via YAC.

## Preparations

Please use the script `setup_venv.sh` to create a virtual environment for
developing and running this dummy.

## Running

Activate the `venv` by running `source .venv/bin/activate`.

### Without coupling

```sh
python3 src/dummy.py --matlab-mesh examples/dem_and_mask.mat
```

### With coupling

```sh
python3 src/dummy.py --matlab-mesh examples/dem_and_mask.mat --couple-to-elmer-ice --couple-to-icon-atmo
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