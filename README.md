# Generation of fractional Gaussian noise and fractional Brownian motion
This is an implementation to generate fractional Gaussian noise (FGN), and by extension calculating the cumulative sum of the FGN gives fractional Brownian motion (FBM).

## A bit of theory
Fractional Brownian motion (FBM) are a family of Gaussian random functions that are self-similar and the increments of these processes are stationary. The increments of these processes are known as fractional Gaussian noise. These processes are of course Gaussian with known autocorrelation function
$$
\rho(k) = \frac{1}{2} [(k + 1)^{2\mathcal{H}} – 2k^{2\mathcal{H}} + |k - 1|^{2\mathcal{H}}]
$$
Where $\mathcal{H}$ is the hurst exponent and we notice for $\mathcal{H}=1/2$ we have zereo correlation, meaning we have uncorrelated (white) noise. For $\mathcal{H}> 1/2$ the increments are persistent and for $\mathcal{H}<1/2$ the increments are anit-persistent.


## Methods
The methods implemented to generate the fractional Gaussian noise is the *Davies-Harte* method and the *Hosking* method.

### The Davies-Harte method
The Davies-Harte method was first outlined in the paper __Tests for hurst effekt__ by R. B. Davies and D. S. Harte [1]. The method was later improved by A. T. A. Wood and G. Chan [2]. This algorithm outlined in this paper is implemented here. The method is based on creating a longer vector than in the original method from a circulant covariance matrix and then selecting the subset of that vector as output. It can be shown that the eigenvalues of a circulant matrix are the Fourier modes.

### The Hosking method
This method is recursive that generates a stationary time series with a gaussian marginal distribution and with the given correlational structure $\rho(k)$.


Implemented methods to generate the FGN is for the moment the Davies-Harte method and the hosking method reached by setting the *method* argument
```python
method = 'davies-harte'
```
or
```python
method = 'hosking'
```

## Usage
The syntax for generating the FGN/FBM time series is

``` python
# import functions
from generating_fbm_fgn import FBM, FGN


# to generate an FBM time series with hurst exponent hurst = 0.7 with length 10 000 datapoints
time_series_fbm = FBM(N = 10000, hurst = 0.7, method='davies-harte')

# to generate an FGN time series with hurst exponent hurst = 0.7 with length 10 000 datapoints
time_series_fGN = FGN(N = 10000, hurst = 0.7, method='davies-harte')
```
or with the __hosking__ method

``` python
# to generate an FBM time series with hurst exponent hurst = 0.7 with length 10 000 datapoints
time_series_fbm = FBM(N = 10000, hurst = 0.7, method='hosking')

# to generate an FGN time series with hurst exponent hurst = 0.7 with length 10 000 datapoints
time_series_fGN = FGN(N = 10000, hurst = 0.7, method='hosking')
```
### NOTE
The hosking method is recursively generated and since python is not exactly known to be a fast language the hosking method can get really slow if the time series you want to generate has some length to it. If the time series has length >2^15 the *davies-harte* method is recommended.

Even though the *hosking* method is slow, that method will not fail. The *davies-harte* method will fail when the simulation length and the hurst exponent are in such combination to obtain negative eigenvalues of the circulant correlation matrix used in the method.  This usually occurs for really high values of the hurst exponent (>0.95) and long simulation time series.  


## References
[1]   R. B. Daveis, D. S. Harte. __Tests for Hurst effect__, (1987). https://doi.org/10.2307/2336024
[2]   A. T. A. Wood, G. Chan. __Simulation of Stationary Gaussian Processes in [0,1]$^{d}$__, (1994). https://doi.org/10.2307/1390903
