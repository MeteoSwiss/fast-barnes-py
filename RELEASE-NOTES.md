<img src="doc/images/InterpolationStrip.png"/>

# Release Notes

These are the release notes for the `fast-barnes-py` Python library.


## 2.0 Release (2024-11-15)

### New Features

- In addition to 2D grids, fast-barnes-py now also supports fast Barnes interpolation for 1D grids and 3D grids.
- The Gaussian width parameter `sigma` and the grid size `step` can now be specified separately for each dimension.
- Exposure of the `max_dist` parameter, which specifies the distance at which a grid point is considered too distant from the sample points and is set to NaN. Before, this parameter was only used internally and set to 3.5 *`sigma`.

### Breaking Change

The order of the dimensions in the `size` parameter has been changed from (y, x) to (x, y) &ndash; or in case of 3D from (z, y, x) to (x, y, z) &ndash; and thus complies with the order of dimensions of all other parameters such as `x0` or `sigma`, `step`, if they are specified as multidimensional arguments.

**Note:** The index mapping of the resulting interpolation array is [y, x] in case of 2D grids and [z, y, x] for 3D grids.
As numpy by default arranges arrays in row-major order, accesses to (y, x)-slices of the array &ndash; which are most frequently used in geo sciences &ndash; in this way can be processed very efficiently because the corresponding array elements are stored in a consecutive block of memory. 

### Bug Fixes / Improvements

- Improved and extensive validity checks of function arguments
  - e.g. check for consistent dimensionality of arguments.
- In cases where the size of the resulting rectangular kernel would be greater than or equal to the grid size (in any direction), a `RuntimeError` is raised, as otherwise the computed interpolation would become invalid.
- Internally working on a copy of sample value array `val` in order to prevent side effects in down-stream computations.
- Replaced `numpy.NaN` values with `numpy.nan` and thus established numpy 2.0 compatibility.


## 1.0 Release (2023-02-23)

Initial implementation of Barnes implementation on 2D grids as described in the corresponding [paper published in the GMD journal](https://gmd.copernicus.org/articles/16/1697/2023/gmd-16-1697-2023.pdf).

