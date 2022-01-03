# Generation of fractional Gaussian noise and fractional Brownian motion
This is an implementation to generate fractional Gaussian noise (FGN), and by extension calculating the cumulative sum of the FGN gives fractional Brownian motion (FBM).

## Methods
Implemented methods to generate the FGN is for the moment only the Davies-Harte method

## Usage
The syntax for generating the FGN/FBM time series is

``` python
# to generate an FBM time series with hurst exponent hurst = 0.7 with length 10 000 datapoints
time_series_fbm = FBM(N = 10000, hurst = 0.7, method='davies-harte')

# to generate an FGN time series with hurst exponent hurst = 0.7 with length 10 000 datapoints
time_series_fGN = FGN(N = 10000, hurst = 0.7, method='davies-harte')
```

## Possible future work
 - [ ] Implement more method
 - [ ] Clean up the code
 - [ ] Finish up the documentation for the code
