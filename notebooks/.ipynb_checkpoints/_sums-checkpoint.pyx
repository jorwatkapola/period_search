# cython structure based on Pim Schellart's (2010 - 2011) lomb scargle code for scipy

"""Tools for spectral analysis of unequally sampled signals."""

import numpy as np
cimport numpy as np
cimport cython

__all__ = ['_wwt_sums']


cdef extern from "math.h":
    double cos(double)
    double sin(double)
    double exp(double)

@cython.boundscheck(False)
def _wwt_sums(np.ndarray[np.float64_t, ndim=1] x,
                np.ndarray[np.float64_t, ndim=1] y,
                np.ndarray[np.float64_t, ndim=1] freqs,
                np.ndarray[np.float64_t, ndim=1] taus,
                np.float64_t decay):
    """
    _lombscargle(x, y, freqs, taus)

    Computes the Lomb-Scargle periodogram.

    Parameters
    ----------
    x : array_like
        Sample times.
    y : array_like
        Measurement values (must be registered so the mean is zero).
    freqs : array_like
        Angular frequencies for output periodogram.

    Returns
    -------
    pgram : array_like
        Lomb-Scargle periodogram.

    Raises
    ------
    ValueError
        If the input arrays `x` and `y` do not have the same shape.

    See also
    --------
    lombscargle

    """

    # Check input sizes
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays do not have the same size.")

    # Create empty array for output
    Yvector_ar = np.empty((taus.shape[0], freqs.shape[0], 3), dtype=np.float64)
    Smatrix_ar = np.empty((taus.shape[0], freqs.shape[0], 3, 3), dtype=np.float64)
    weight2_ar = np.empty((taus.shape[0], freqs.shape[0]), dtype=np.float64)
    varw_ar = np.empty((taus.shape[0], freqs.shape[0]), dtype=np.float64)

    # Local variables
    cdef int kstart
    cdef Py_ssize_t idat, ifreq, itau
    cdef double omega, z, weight, c, s, wc, ws, wy
    cdef double weight2, varw
    cdef double Smatrix00, Smatrix01, Smatrix02, Smatrix11, Smatrix12, Smatrix22
    cdef double Yvector0, Yvector1, Yvector2
    
    for itau in range(taus.shape[0]):
        
        xstart = 0

        for ifreq in range(freqs.shape[0]):
            
            omega = 2.0 * np.pi * freqs[ifreq]
            
            Smatrix00 = 0.
            Smatrix01 = 0.
            Smatrix02 = 0.
            Smatrix11 = 0.
            Smatrix12 = 0.
            Smatrix22 = 0.

            Yvector0 = 0.
            Yvector1 = 0.
            Yvector2 = 0.
            
            weight2 = 0.
            varw = 0.

            for idat in range(xstart, x.shape[0]):
                
                z = omega * (x[idat] - taus[itau])
                weight = exp(-1.0 * decay * z * z) 
                
                if weight > 1e-9:
                
                    c = cos(z)
                    s = sin(z)
                    wc = weight * c
                    ws = weight * s
                    wy = weight * y[idat]
                    
                    Smatrix00 += weight
                    Smatrix01 += wc
                    Smatrix02 += ws
                    Smatrix11 += wc * c
                    Smatrix12 += wc * s
                    Smatrix22 += ws * s
                    
                    Yvector0 += wy
                    Yvector1 += weight * y[idat] * c
                    Yvector2 += weight * y[idat] * s
                    
                    weight2 += weight * weight
                    varw += wy * y[idat]
                    
                elif z > 0.:
                    break
                else:
                    xstart += 1

                    
            Smatrix_ar[itau, ifreq, 0, 0] = Smatrix00
            Smatrix_ar[itau, ifreq, 0, 1] = Smatrix01
            Smatrix_ar[itau, ifreq, 0, 2] = Smatrix02
            Smatrix_ar[itau, ifreq, 1, 1] = Smatrix11
            Smatrix_ar[itau, ifreq, 1, 2] = Smatrix12
            Smatrix_ar[itau, ifreq, 2, 2] = Smatrix22
            
            Yvector_ar[itau, ifreq, 0] = Yvector0
            Yvector_ar[itau, ifreq, 1] = Yvector1
            Yvector_ar[itau, ifreq, 2] = Yvector2
            
            weight2_ar[itau, ifreq] = weight2
            varw_ar[itau, ifreq] = varw

    return Smatrix_ar, Yvector_ar, weight2_ar, varw_ar
