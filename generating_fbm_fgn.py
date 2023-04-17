'''
The most common method is the Davies-Harte algorithm described in Tests for
Hurst effect, and more recently improved by Wood and Chan

This implementation will follow that of Wood and Chan with the same nomentcalture

The Hosking method is also implemented, following the paper by J. R. M Hosking
'''

import numpy as np
import math
from numba import njit



def FBM(N: int, hurst: float, method: str):
    '''
    Calculate the fractional Brownian motion from the generated fractional
    Gaussian noise process, generated to a function-call to the FGN function.
    The FBM is the cumulative sum of the FGN.

    Parameters
    ----------
    N : int
        Length of simulation.
    hurst : float
        Hurst exponent of the FGN process.
    method : str
        Method for generating the fractional Gaussian noise.

    Returns
    -------
    FBM : array
        Cumulative sum of the FGN simulation, a fractional Brownian 
        motion time series
    '''

    fgn = FGN(N, hurst, method)
    return np.cumsum(fgn)


def FGN(N: int, hurst: float, method: str):
    '''
    Generating the fractional Gaussian noise time series using a determined
    method for generating the time series.

    The implemented method: Davies-Harte method

    Special case for Hurst exponent 1/2 where the FGN is uncorrelated normally
    distributed noise

    Parameters
    ----------
    N : int
        Length of simulation.
    hurst : float
        Hurst exponent of the FGN process.
    method : str
        Method for generating the fractional Gaussian noise.

    Returns
    -------
    fgn : array
        The fractional Gaussian noise time series

    '''
    assert 0 < hurst < 1, 'Hurst exponent has to be in the range (0, 1)'
    assert method in ['davies-harte', 'hosking'], 'Unsupported method. Use "davies-harte" or "hosking"'
    if hurst == 0.5:
        #For Hurst exponent == 0.5 the process is just a normal white noise
        fgn = np.random.normal(0, 1, size = N)
        return fgn

    else:
        if method  == 'davies-harte':
            fgn = fgn_DH(N, hurst)
            return fgn
        if method == 'hosking':
            fgn = fgn_hosking(N, hurst)
            return fgn

@njit(cache=True)
def Autocov_func(k: int, hurst: float):
    '''
    The Autocovariance function which defines the fractional Gaussian noise
    Returns the covariance between two datapoints "k" steps apart

    Parameters
    ----------
    k : int
        Lag used in the calculation of the auto-correlation function.
    hurst : float
        HUrst exponent used when generating the FGN time series.

    Returns
    -------
    ACF : float
        Auto-correlation at given lag and Hurst exponent
    '''
    return 0.5*np.abs(k + 1)**(2*hurst) + 0.5*np.abs(k - 1)**(2*hurst) - np.abs(k)**(2*hurst)


def fgn_DH(N: int, hurst: float):
    '''
    Implementation of the Davies-Harte algorithm to generate the fractional
    Gaussian noise

    Parameters
    ----------
    N : int
        Length of the simulation.
    hurst : float
        Hurst exponent used when generating the FGN time series.

    Raises
    ------
    ValueError
        Raised when negative eigenvalues of the circulant matrix are encountered 

    Returns
    -------
    fgn : array
        Scaled fractional Gaussian noise time series.

    '''

    g = math.ceil(np.log2(2*N))
    m = int(2**g)

    # generate row in circulant matrix C
    comp_row = [Autocov_func(x, hurst) for x in range(1, int(m*0.5))]
    reversed_row = comp_row[::-1]
    row = [Autocov_func(0, hurst)] + comp_row + [Autocov_func(int(m*0.5), hurst)] + reversed_row
    # print('length of row circulant matrix: {}'.format(len(row)))


    try:
        eigenvalues = np.fft.fft(row).real
        if np.any([eigenvalue < 0 for eigenvalue in eigenvalues]):
            raise ValueError

    except ValueError:
        print('Error: Encountered negative eigenvalue again. Switching to an approximate solution...')
        
        # NOTE: You can increase the number "g" to find a time series length which does not
        # give you negative eigenvalues, this has, however, not been necessary thus far. 
  

        # Removing the elements corresponding to the negative eigenvalues in the circulant matrix
        index = [n for n in range(len(eigenvalues)) if eigenvalues[n] < 0]
        print('Number of negative eigenvalues: {}'.format(len(index)))
        c1 = np.sum(eigenvalues) # summing over all eigenvalues
        for idx in index:
            eigenvalues[idx] = 0.0
        c2 = np.sum(eigenvalues) # summing over all positive eigenvalues

        # Setting all row elements corresponding to a negiative eigenvalue to zero
        for idx in index:
            row[idx] = 0.0

        # Adding correction to the circulant matrix row.
        row = (c1 / c2)**2 * np.array(row)
        eigenvalues = np.fft.fft(row)


    #Generating two independent standard normal random variables for use later
    U = np.random.normal(0, 1, int(m))
    V = np.random.normal(0, 1, int(m))

    a = np.zeros(m, dtype=complex)
    a[0] = np.sqrt(eigenvalues[0]/m) * U[0]
    a[1:int(0.5*m)] = np.sqrt(eigenvalues[1:int(0.5*m)] / (2 * m)) * (U[1:int(0.5*m)] + 1j*V[1:int(0.5*m)])
    a[int(0.5*m)] = np.sqrt(eigenvalues[int(0.5*m)] / m) * V[int(0.5*m)]
    a[int(0.5*m)+1:] = np.sqrt(eigenvalues[int(0.5*m)+1:] / (2 * m)) * (U[int(0.5*m)+1:] - 1j*V[int(0.5*m)+1:])

    x = np.fft.fft(a)
    fgn = x[:N].real

    scale = 1 #(1.0/N) ** hurst
    # This is to scale the normal random variables to fit with the stepsize
    # of the time increments of the simulation. These should be
    # sqrt(delta(t)) * N(0,1) for a standard brownian motion
    # it is the stepsize to the power of the hurst exponent
    return fgn * scale

@njit(cache=True)
def fgn_hosking(N: int, hurst: float):
    '''
    Implementation of the Hosking method to generate a fractional Gaussian
    noise time series

    Parameters
    ----------
    N : int
        Length of the simulation.
    hurst : float
        Hurst exponent used when generating the FGN time series.


    Returns
    -------
    fgn : array
        Scaled fractional Gaussian noise time series.

    '''
    

    # Allocate resulting time series
    fgn = np.zeros(N)
    # partial correlation coefficients
    phi_ii = np.zeros(N)
    # partial linear regression coefficients
    phi_ij = np.zeros(N)

    # Generate standard normal distribution
    normal = np.random.normal(0, 1, N)

    # Initial values
    fgn[0] = normal[0]
    V = 1
    phi_ii[0] = 0

    for i in range(1, N):
        phi_ii[i - 1] = Autocov_func(i, hurst)

        for j in range(i - 1):
            phi_ij[j] = phi_ii[j]
            phi_ii[i - 1] -= phi_ij[j] * Autocov_func(i - j - 1, hurst)

        phi_ii[i - 1] /= V

        for j in range(i - 1):
            phi_ii[j] = phi_ij[j] - phi_ii[i - 1]*phi_ij[i - j - 2]
        V *= 1 - phi_ii[i - 1]**2

        for j in range(i):
            fgn[i] += phi_ii[j] * fgn[i - j - 1]

        fgn[i] += np.sqrt(V) * normal[i]

    scale = (1.0/N) ** hurst
    return fgn * scale

# End of File
