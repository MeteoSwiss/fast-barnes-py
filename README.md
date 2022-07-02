# Fast Barnes Interpolation
This repository provides a Java implementation of the formal algorithms for fast Barnes interpolation as presented in the corresponding paper.

Barnes interpolation is a method that is widely used in geospatial sciences like meteorology to remodel data values recorded at irregularly distributed points into a representative analytical field.
It is defined as

<img src="doc\images\BarnesInterpolDef.png" width="185"/>

with Gaussian weights

<img src="doc\images\GaussianWeights.png" width="175"/>

Naive computation of Barnes interpolation leads to an algorithmic complexity of O(N x W x H), where N is the number of sample points and W x H the size of the underlying grid.  
As pointed out in the paper, for sufficiently large n (in general in the range from 3 to 6) a good approximation of Barnes interpolation with a reduced complexity O(N + W x H) can be obtained by the convolutional expression

<img src="doc\images\BarnesInterpolConvolExpr.png" width="343"/>

where &delta; is the Dirac impulse function and r(.) an elementary rectangular function of a specific length that depends on &sigma; and n.

The class `ConvolBarnesInterpol` implements the basic convolutional algorithm as described in chapter 4, while `OptConvolBarnesInterpol` provides its optimized version treated in chapter 5.4.  
The two classes `Main` and `MainS2` allow the computation of Barnes interpolation for various setups as also used in the paper.
If the boolean variable `writeResultFile` is set to true, the resulting data is written to a file and can be subsequently visualized by the auxiliary Python program `Paper_Map` in the tools folder.
The five `Measurement` classes perform the time measurements and were used to assemble the data shown in Table 1 to 5.
