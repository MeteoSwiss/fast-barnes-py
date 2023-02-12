# Fast Barnes Interpolation
This repository provides a Python implementation of the formal algorithms for fast Barnes interpolation as presented in the corresponding paper ([preprint available at GMD](https://gmd.copernicus.org/preprints/gmd-2022-116/gmd-2022-116.pdf)).

Barnes interpolation is a method that is widely used in geospatial sciences like meteorology to remodel data values recorded at irregularly distributed points into a representative analytical field.
It is defined as

<img src="doc\images\BarnesInterpolDef.png" width="185"/>

with Gaussian weights

<img src="doc\images\GaussianWeights.png" width="175"/>

Naive computation of Barnes interpolation leads to an algorithmic complexity of O(N x W x H), where N is the number of sample points and W x H the size of the underlying grid.  
As pointed out in the paper, for sufficiently large n (in general in the range from 3 to 6) a good approximation of Barnes interpolation with a reduced complexity O(N + W x H) can be obtained by the convolutional expression

<img src="doc\images\BarnesInterpolConvolExpr.png" width="343"/>

where &delta; is the Dirac impulse function and r(.) an elementary rectangular function of a specific length that depends on &sigma; and n.

The module `interpolation` implements the Barnes interpolation algorithms using the Euclidean distance metric, as described in chapter 4 and 5.4 of the paper.
The Barnes interpolation algorithms that use spherical distance metric on the sphere S^2, as outlined in chapter 5.5, are implemented im module `interpolationS2`.

The directory `demo` provides Python programs that reproduce the figures and the tables shown in the paper.
The corresponding scripts `figure*.py` and `timing*.py` are parameterless and can be executed directly from this directory.
In order to do this successfully, take care to have all packages listed in the `requirements.txt` file installed.