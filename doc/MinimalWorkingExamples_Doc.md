<img src="https://github.com/MeteoSwiss/fast-barnes-py/blob/main/doc/images/InterpolationStrip.png?raw=true"/>

# Minimal Working Examples

This page includes minimal working examples for the application of fast Barnes interpolation to data of different dimensions.

- [Example for 2-Dimensional Data](#example-for-2-dimensional-data)  
- [Example for 1-Dimensional Data](#example-for-1-dimensional-data)

&nbsp;

<a id="example-for-2-dimensional-data"></a>
###  Example for 2-Dimensional Data

The code below demonstrates how Barnes interpolation can be computed given a few sample points of mean sea level pressure values located over the British islands.

```
import numpy as np

# definition of 50 sample points with longitude, latitude and mean sea level pressure QFF
input_data = np.asarray([
    [-3.73,56.33,995.1], [2.64,47.05,1012.5], [-8.40,47.50,1011.3], [2.94,54.33,1006.0],
    [-2.90,49.90,1006.3], [-8.98,53.72,1002.1], [1.20,58.60,1002.6], [1.60,50.73,1009.1],
    [-7.38,57.36,997.7], [-1.25,53.01,1000.4], [-4.74,52.79,998.4], [-0.61,47.48,1013.0],
    [-6.10,50.10,1004.3], [-6.46,54.87,996.4], [-3.22,47.29,1012.8], [-1.60,55.42,996.6],
    [2.30,56.60,1004.5], [1.12,52.95,1003.6], [-0.90,57.80,999.9], [-7.90,51.40,1002.6],
    [-0.70,50.10,1007.5], [2.53,49.02,1010.8], [-5.06,48.47,1008.5], [-3.10,53.60,997.5],
    [-5.63,57.86,997.8], [-6.90,52.85,1000.9], [-4.15,51.09,1002.6], [-1.99,51.50,1002.7],
    [1.21,47.68,1011.7], [-5.70,56.30,995.5], [-1.98,53.13,998.5], [1.09,49.93,1009.0],
    [1.72,58.42,1002.9], [-6.30,52.30,999.4], [0.70,57.70,1001.9], [-3.50,53.60,995.9],
    [1.38,48.06,1011.6], [-4.37,51.71,1001.1], [-3.09,58.45,998.5], [2.00,56.40,1003.9],
    [1.90,57.00,1003.3], [0.45,51.90,1004.9], [-8.25,51.80,1002.5], [-1.87,53.81,997.4],
    [-2.38,55.71,995.1], [-4.01,54.80,992.1], [0.88,53.37,1002.6], [-1.69,51.86,1002.1],
    [-4.57,52.14,999.6], [-0.20,58.40,1001.1],
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

&nbsp;

<a id="example-for-1-dimensional-data"></a>
###  Example for 1-Dimensional Data

Barnes interpolation can also be applied to one-dimensional data, as shown below using temperature data that were recorded at irregular time intervals.

```
import numpy as np

# definition of 94 AMDAR samples with observation time, temperature [°C] and
# geopotential [m] on 925 hPa level after a cold front traversal at Frankfurt
# airport "EDDF" from 25.08.2024
input_data = [
    [ "03:15:00", 15.4, 799.0 ], [ "03:35:00", 12.8, 793.0 ], [ "04:51:00", 11.5, 807.0 ],
    [ "04:58:00", 10.7, 805.0 ], [ "05:03:00", 11.9, 807.0 ], [ "05:05:00", 11.3, 805.0 ],
    [ "05:12:00", 12.7, 807.0 ], [ "05:14:00", 11.8, 806.0 ], [ "05:23:00", 12.6, 816.0 ],
    [ "05:24:00", 11.3, 813.0 ], [ "05:32:04", 11.6, 818.0 ], [ "05:33:00", 12.9, 818.0 ],
    [ "05:36:00", 11.8, 815.0 ], [ "05:45:00", 10.9, 822.0 ], [ "05:47:00", 11.5, 823.0 ],
    [ "05:53:00", 10.2, 820.0 ], [ "06:05:37", 10.9, 823.0 ], [ "06:12:00", 10.3, 820.0 ],
    [ "06:19:00",  9.8, 819.0 ], [ "06:20:00", 10.3, 821.0 ], [ "06:21:39", 11.6, 825.0 ],
    [ "06:30:00", 10.9, 823.0 ], [ "06:33:35", 10.7, 824.0 ], [ "06:59:00", 10.8, 829.0 ],
    [ "07:09:00", 10.2, 829.0 ], [ "07:11:00", 10.0, 830.0 ], [ "07:21:00", 10.6, 831.0 ],
    [ "07:29:00",  9.6, 830.0 ], [ "08:18:00", 10.3, 839.0 ], [ "08:20:00", 10.5, 840.0 ],
    [ "08:22:00", 10.6, 839.0 ], [ "08:36:00", 10.4, 838.0 ], [ "08:39:00", 11.2, 842.0 ],
    [ "08:52:00", 11.1, 840.0 ], [ "09:00:16", 11.7, 842.0 ], [ "09:14:00", 10.4, 841.0 ],
    [ "09:24:00", 12.6, 845.0 ], [ "09:39:00", 10.8, 839.0 ], [ "09:57:00", 11.3, 841.0 ],
    [ "10:02:00", 11.4, 841.0 ], [ "10:09:00", 11.0, 840.0 ], [ "10:19:00", 11.7, 841.0 ],
    [ "10:33:00", 11.2, 840.0 ], [ "10:52:00", 11.9, 843.0 ], [ "10:53:00", 12.5, 845.0 ],
    [ "11:01:00", 12.7, 845.0 ], [ "11:03:00", 11.6, 843.0 ], [ "11:33:04", 11.8, 843.0 ],
    [ "11:35:00", 12.3, 844.0 ], [ "11:36:00", 13.1, 845.0 ], [ "12:13:00", 12.3, 844.0 ],
    [ "12:17:00", 13.4, 846.0 ], [ "12:49:00", 12.8, 846.0 ], [ "12:54:00", 12.6, 844.0 ],
    [ "13:08:00", 13.5, 847.0 ], [ "13:22:00", 14.6, 849.0 ], [ "13:30:00", 12.8, 846.0 ],
    [ "13:35:00", 14.1, 848.0 ], [ "13:44:02", 13.8, 848.0 ], [ "13:54:47", 14.1, 850.0 ],
    [ "14:05:00", 13.7, 848.0 ], [ "14:11:00", 13.8, 849.0 ], [ "14:17:00", 13.3, 848.0 ],
    [ "14:24:00", 13.6, 849.0 ], [ "14:31:00", 13.9, 849.0 ], [ "14:39:00", 15.1, 851.0 ],
    [ "14:54:00", 13.5, 847.0 ], [ "15:20:00", 13.9, 848.0 ], [ "15:25:00", 14.9, 850.0 ],
    [ "15:26:00", 14.0, 848.0 ], [ "15:28:00", 14.9, 851.0 ], [ "15:29:00", 14.6, 849.0 ],
    [ "15:44:00", 13.8, 849.0 ], [ "15:46:00", 14.2, 849.0 ], [ "16:19:06", 14.8, 851.0 ],
    [ "16:36:00", 13.9, 847.0 ], [ "16:39:00", 13.9, 848.0 ], [ "16:46:00", 14.2, 849.0 ],
    [ "16:48:00", 13.6, 847.0 ], [ "17:08:00", 14.1, 847.0 ], [ "17:29:00", 14.2, 848.0 ],
    [ "17:51:26", 15.8, 853.0 ], [ "18:04:00", 13.9, 847.0 ], [ "18:15:05", 15.2, 851.0 ],
    [ "18:19:00", 13.9, 847.0 ], [ "18:25:00", 15.4, 851.0 ], [ "18:34:00", 15.1, 849.0 ],
    [ "18:54:00", 13.5, 846.0 ], [ "18:58:00", 15.1, 850.0 ], [ "18:58:23", 13.9, 847.0 ],
    [ "19:16:00", 14.1, 847.0 ], [ "19:32:00", 14.0, 846.0 ], [ "20:14:00", 14.5, 856.0 ],
    [ "20:44:00", 13.3, 853.0 ],
]

def seconds_of_day(timestr):
    """ Translates a time String 'HH:mm:ss' into the equivalent number of seconds since 0:00. """
    arr = timestr.split(':')
    return (int(arr[0])*60 + int(arr[1]))*60 + int(arr[2])

# read data
input_data = [ [seconds_of_day(item[0]), item[1] ] for item in input_data]
input_data = np.asarray(input_data)

# extract time and temperature arrays
time_values = input_data[:, 0]
temp_values = input_data[:, 1]
```

The target time grid is specified to encompass a entire day with a grid point distance of 30 seconds.
The time array and the corresponding temperature array are passed with a reasonable Gaussian width parameter $\sigma$ = 40 min to the `interpolation.barnes()` method.

```
# definition of a time grid (with unit seconds) starting at 00:00 until 24:00 hPa with
# steps of 30 seconds
step = 30.0
x0 = 0.0
size = int(86400 / step + 1)

# calculate Barnes interpolation
from fastbarnes import interpolation
sigma = 2400.0
temp_arr = interpolation.barnes(time_values, temp_values, sigma, x0, step, size)
```

The resulting temperature array containing the interpolated values can then be plotted by means of matplotlib, for example.

```
# plot interpolation of temperature and scattered input sample temperature values
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 3.5), dpi=150)
plt.grid(visible=True)
plt.xlim(7200.0, 79200.0)
plt.xticks([10800, 21600, 32400, 43200, 54000, 64800, 75600],
    ['03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00'])
plt.ylim(9.3, 15.7)

plt.title("Frankfurt Airport AMDAR Reports from 25.08.2024")
plt.xlabel("Day Time [UTC]")
plt.ylabel("Temperature [°C]\non 925hPa level")

grid = np.arange(x0, x0+size*step, step)
print(temp_arr)
print(grid)
plt.plot(grid, temp_arr)
plt.scatter(time_values, temp_values, color='red', s=20, marker='.')
print(time_values)
print(temp_values)

plt.tight_layout()
plt.show()
```

The result of the example code above is given by the following graphic.

<img src="https://github.com/MeteoSwiss/fast-barnes-py/blob/main/doc/images/AmdarTemperature.png?raw=true" width="1000"/>
