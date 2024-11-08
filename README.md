<img src="doc/images/InterpolationStrip.png"/>

# Fast Barnes Interpolation
This repository provides a Python implementation of the formal algorithms for fast Barnes interpolation as presented in the corresponding [paper published in the GMD journal](https://gmd.copernicus.org/articles/16/1697/2023/gmd-16-1697-2023.pdf).  
In addition to Barnes interpolation for 2-dimensional applications, this implementation now also supports the interpolation of 1-dimensional and 3-dimensional data.

### Mathematical Background

Barnes interpolation is a method that is widely used in geospatial sciences like meteorology to remodel data values $f_k$ recorded at irregularly distributed points $\pmb{x}_k$ into a representative analytical field $f(\pmb{x})$.
It is defined as

<img src="doc\images\BarnesInterpolDef.png" width="185"/>

with Gaussian weights

<img src="doc\images\GaussianWeights.png" width="175"/>

for a specific Gaussian width parameter $\sigma$.

Naive computation of Barnes interpolation leads to an algorithmic complexity of $\mathcal{O}(N \cdot W \cdot H)$, where $N$ is the number of sample points and $W \times H$ the size of the underlying grid.  
As shown in the paper, for sufficiently large $n$ (in general in the range from 3 to 6) a good approximation of Barnes interpolation with a reduced complexity $\mathcal{O}(N + W \cdot H)$ can be obtained by the convolutional expression

<img src="doc\images\BarnesInterpolConvolExpr.png" width="343"/>

where $\delta_{\pmb{x}_k}$ is the Dirac impulse function at location $\pmb{x}_k$ and $r_n(.)$ an elementary rectangular function of a specific length that depends on $\sigma$ and $n$.

### Repository Structure

The module `interpolation` implements the Barnes interpolation algorithms using the Euclidean distance metric, as described in chapter 4 and 5.4 of the paper.
The Barnes interpolation algorithms that use spherical distance metric on the sphere $\mathcal{S}^2$, as outlined in chapter 5.5, are implemented im module `interpolationS2`.
However, be aware here that the supported geographical domain and projection - as in the paper - is currently fixed to the European latitudes and Lambert conformal projection and cannot be freely chosen.
These algorithms are also available as fast-barnes-py package on PyPI.

The directory `demo` provides Python scripts that reproduce the figures and the tables shown in the paper.
In order to execute them you can follow [these instructions](./doc/ReproduceResults_Doc.md).

If you want to find out how to use the code, you best read through the provided [minimal working examples](./doc/PyPI_Doc.md).

