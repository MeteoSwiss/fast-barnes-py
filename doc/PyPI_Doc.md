# Fast Barnes Interpolation
This Python package provides an implementation of the formal algorithms for fast Barnes interpolation as presented in the corresponding [paper published in the GMD journal](https://gmd.copernicus.org/articles/16/1697/2023/gmd-16-1697-2023.pdf).

Barnes interpolation is a method that is widely used in geospatial sciences like meteorology to remodel data values recorded at irregularly distributed points into a representative analytical field.

Naive computation of Barnes interpolation leads to an algorithmic complexity of O(N x W x H), where N is the number of sample points and W x H the size of the underlying grid.  
As shown in the paper, a good approximation of Barnes interpolation with a drastically reduced algorithmic complexity O(N + W x H) can be obtained by calculating a convolutional expression.

### Minimal Working Example

The code below demonstrates how Barnes interpolation can be computed given a few sample points of mean sea level pressure values located over the British islands.

```
import numpy as np

# definition of 50 sample points with longitude, latitude and mean sea level pressure QFF
input_data = np.asarray([
    [-3.73,56.33,995.1], [2.64,47.05,1012.5], [-8.40,47.50,1011.3], [2.94,54.33,1006.0], [-2.90,49.90,1006.3],
    [-8.98,53.72,1002.1], [1.20,58.60,1002.6], [1.60,50.73,1009.1], [-7.38,57.36,997.7], [-1.25,53.01,1000.4],
    [-4.74,52.79,998.4], [-0.61,47.48,1013.0], [-6.10,50.10,1004.3], [-6.46,54.87,996.4], [-3.22,47.29,1012.8],
    [-1.60,55.42,996.6], [2.30,56.60,1004.5], [1.12,52.95,1003.6], [-0.90,57.80,999.9], [-7.90,51.40,1002.6],
    [-0.70,50.10,1007.5], [2.53,49.02,1010.8], [-5.06,48.47,1008.5], [-3.10,53.60,997.5], [-5.63,57.86,997.8],
    [-6.90,52.85,1000.9], [-4.15,51.09,1002.6], [-1.99,51.50,1002.7], [1.21,47.68,1011.7], [-5.70,56.30,995.5],
    [-1.98,53.13,998.5], [1.09,49.93,1009.0], [1.72,58.42,1002.9], [-6.30,52.30,999.4], [0.70,57.70,1001.9],
    [-3.50,53.60,995.9], [1.38,48.06,1011.6], [-4.37,51.71,1001.1], [-3.09,58.45,998.5], [2.00,56.40,1003.9],
    [1.90,57.00,1003.3], [0.45,51.90,1004.9], [-8.25,51.80,1002.5], [-1.87,53.81,997.4], [-2.38,55.71,995.1],
    [-4.01,54.80,992.1], [0.88,53.37,1002.6], [-1.69,51.86,1002.1], [-4.57,52.14,999.6], [-0.20,58.40,1001.1],
])

lon_lat_data = input_data[:, 0:2]
qff_values = input_data[:, 2]
```

When displayed as a scatter plot, the points defined above produce the following chart (code not shown):

<img src="https://github.com/MeteoSwiss/fast-barnes-py/blob/main/doc/images/Samples.png?raw=true" width="500"/>

Now the target grid has to be specified and then the data and the grid are passed with the Gaussian width parameter to the `interpolation.barnes()` method, which returns a representative gridded field. 

```
# definition of a 12° x 12° grid starting at 9°W / 47°N
resolution = 32.0
step = 1.0 / resolution
x0 = np.asarray([-9.0, 47.0], dtype=np.float64)
size = (int(12.0 / step), int(12.0 / step))

# calculate Barnes interpolation
from fastbarnes import interpolation
sigma = 1.0
field = interpolation.barnes(lon_lat_data, qff_values, sigma, x0, step, size)
```

The resulting field can then be further processed, for instance visualized by a matplotlib contour plot

```
# draw graphic with labeled contours and scattered sample points
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 5))
plt.margins(x=0, y=0)

gridX = np.arange(x0[0], x0[0]+size[1]*step, step)
gridY = np.arange(x0[1], x0[1]+size[0]*step, step)
levels = np.arange(976, 1026, 2)
cs = plt.contour(gridX, gridY, field, levels)
plt.clabel(cs, levels[::2], fmt='%d', fontsize=9)

plt.scatter(lon_lat_data[:, 0], lon_lat_data[:, 1], color='red', s=20, marker='.')

plt.show()
```

Note that due to the just-in-time compilation of the underlying code, the first execution of Barnes interpolation takes considerable more time than the succeeding ones.

This yields the subsequent chart:

<img src="https://github.com/MeteoSwiss/fast-barnes-py/blob/main/doc/images/MweIsolines.png?raw=true" width="500"/>

Decorating these isolines with a nice map background and labeling the sample points with their QFF values finally results in this graphics (code not shown):

<img src="https://github.com/MeteoSwiss/fast-barnes-py/blob/main/doc/images/MapIsolines.png?raw=true" width="500"/>
