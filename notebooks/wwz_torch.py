"""
This module provides functions for computing the weighted wavelet z transform over input values.
"""

from typing import List

# noinspection Mypy
import numpy as np
# noinspection Mypy
from joblib import Parallel, delayed
import multiprocessing
import time
import torch


def make_tau(timestamps: np.ndarray,
             time_divisions: int) -> np.ndarray:
    """
    Creates an array of times with given timestamps and time divisions to iterate over in the wwt code.
    :param timestamps: An array with corresponding times for the magnitude (payload).
    :param time_divisions: number of divisions for the new timestamps
    :return: tau
    """

    # Check to see if time_divisions is smaller than timestamps (replace if larger)
    if time_divisions > len(timestamps):
        time_divisions = len(timestamps)
        print('adjusted time_divisions to: ', time_divisions)

    # Make tau
    tau: np.ndarray = np.linspace(timestamps[0], timestamps[-1], time_divisions)

    return tau


def make_freq(freq_low: float,
              freq_high: float,
              freq_steps: float, ) -> np.ndarray:
    """
    Creates an array of frequencies with given low, high, and steps to iterate over in the wwt code.
    :param freq_low: The low end of frequency to cast WWZ
    :param freq_high: The high end of frequency to cast WWZ
    :param freq_steps: The frequency steps for casting WWZ
    :return: freq
    """
    freq: np.ndarray = np.arange(freq_low, freq_high + freq_steps, freq_steps)

    return freq


def make_octave_freq(freq_target: float,
                     freq_low: float,
                     freq_high: float,
                     band_order: float,
                     log_scale_base: float,
                     freq_pseudo_sr: float,
                     largest_tau_window: float,
                     override: bool) -> np.ndarray:
    """
    Creates an array of frequencies based on Milton Garces' Constant-Q standardized Inferno framework (2013)
    Recommend "Constant-Q Gabor Atoms for Sparse Binary Representations of Cyber-Physical Signatures" (Garces 2020)
    :param freq_target: target frequency / frequency of interest
    :param freq_low: desired minimum frequency to be contained
    :param freq_high: desired maximum frequncy to be contained
    :param band_order: octave band order (N => 1) *Recommend N = (1, 3, 6, 12, 24,...)
    :param log_scale_base: logarithmic scale base to create the octaves
    :param freq_pseudo_sr: pseudo sample rate of the data by taking the average
    :param largest_tau_window: largest tau window made from make_tau function
    :param override: Override the freq_low and freq_high restrictions
    :return:
    """
    # Check and fix if the freq_low and freq_high meet requirements [CAN BE OVERRIDE]
    # Calculate j_min / j_max (band number min / max)
    if freq_low <= 2 / largest_tau_window and override is False:
        print('largest data window duration is too small for freq_low... taking lowest possible...')
        # This is taken from equation 69b
        j_min = np.floor(band_order * np.log2(2 / (largest_tau_window * freq_target))) + 1
    else:
        j_min = np.floor(band_order * np.log2(freq_low / freq_target))

    if freq_high >= freq_pseudo_sr / 2 and override is False:
        print('Nyquist Frequency is too small for freq_high... taking largest possible...')
        # This is taken from equation 69a
        j_max = np.ceil(band_order * np.log2(freq_pseudo_sr / (2 * freq_target))) - 1
    else:
        j_max = np.ceil(band_order * np.log2(freq_high / freq_target))

    # Create an array with the j_min to j_max (equation 70)
    band_numbers = np.arange(j_min, j_max + 1)

    # Compute the octave frequency bands (center frequencies) (equation 70)
    freq = freq_target * log_scale_base ** (band_numbers / band_order)

    return freq


# def wwt(timestamps: np.ndarray,
#         magnitudes: np.ndarray,
#         time_divisions: int,
#         freq_params: list,
#         decay_constant: float,
#         method: str = 'linear',
#         parallel: bool = True) -> np.ndarray:
#     """
#     The code is based on G. Foster's FORTRAN
#     code as well as eaydin's python 2.7 code. The code is updated to use numpy methods and allow for float value tau.
#     It returns an array with matrix of new evenly spaced timestamps, frequencies, wwz-power, amplitude, coefficients,
#     and effective number. Specific equations can be found on Grant Foster's "WAVELETS FOR PERIOD ANALYSIS OF UNEVENLY
#     SAMPLED TIME SERIES". Some of the equations are labeled in the code with corresponding numbers.

#     :param timestamps: An array with corresponding times for the magnitude (payload).
#     :param magnitudes: An array with payload values
#     :param time_divisions: number of divisions for the new timestamps
#     :param freq_params: A list containing parameters for making frequency bands to analyze over with given 'method'
#             'linear' -> [freq_low, freq_high, freq_step, override]
#             'octave' -> [freq_tg, freq_low, freq_high, band_order, log_scale_base, override]
#     :param decay_constant: decay constant for the Morlet wavelet (should be <0.02) eq. 1-2
#             c = 1/(2w), the wavelet decays significantly after a single cycle of 2 * pi / w
#     :param method: determines method of creating freq ('linear', 'octave') default 'linear'
#     :param parallel: boolean indicate to use parallel processing or not
#     :return: Tau, Freq, WWZ, AMP, COEF, NEFF in a numpy array
#     """

#     # Starting Weighted Wavelet Z-transform and start timer...
#     print("*** Starting Weighted Wavelet Z-transform ***\n")
#     process_starttime: float = time.time()

#     # Get taus to compute WWZ (referred in paper as "time shift(s)")
#     tau: np.ndarray = make_tau(timestamps, time_divisions)
#     ntau: int = len(tau)

#     # Calculate pseudo sample rate and largest time window to check for requirements
#     freq_pseudo_sr = 1 / np.median(np.diff(timestamps))  # 1 / median period

#     # noinspection PyArgumentList
#     largest_tau_window = tau[1] - tau[0]
#     print('Pseudo sample frequency (median) is ', np.round(freq_pseudo_sr, 3))
#     print('largest tau window is ', np.round(largest_tau_window, 3))

#     # Frequencies to compute WWZ
#     if method == 'linear':
#         freq: np.ndarray = freq_params # edited - directly feed in frequency steps
#         # freq: np.ndarray = make_freq(freq_low=freq_params[0],
#         #                              freq_high=freq_params[1],
#         #                              freq_steps=freq_params[2])
#         nfreq: int = len(freq)

#     elif method == 'octave':
#         freq = make_octave_freq(freq_target=freq_params[0],
#                                 freq_low=freq_params[1],
#                                 freq_high=freq_params[2],
#                                 band_order=freq_params[3],
#                                 log_scale_base=freq_params[4],
#                                 freq_pseudo_sr=freq_pseudo_sr,
#                                 largest_tau_window=largest_tau_window,
#                                 override=freq_params[5])
#         nfreq = len(freq)

#     # Get number of data from timestamps
#     numdat: int = len(timestamps)

#     # Get number of CPU cores on current device (used for parallel)
#     num_cores = multiprocessing.cpu_count()

#     # WWT Stars Here
#     def tau_loop(dtau):
#         """
#         Replaced the for loop of the taus ("time shifts") for parallel processing.
#         Comments include connections to the formula given in the Foster96.
#         :param dtau: one of the taus being iterated
#         :return: a single entry of Tau, Freq, WWZ, AMP, COEF, NEFF corresponding to dtau
#         """
#         # Initialize the outputs for each iteration
#         index: int = 0
#         output: np.ndarray = np.empty((len(freq), 6))
#         nstart: int = 1
#         dvarw: float = 0.0

#         # loop over each interested "frequency" over the "time shifts"
#         for dfreq in freq:
#             # Initialize a vector (3) and matrix (3,3) and dweight2 (sum of dweight**2)
#             dvec: np.ndarray = np.zeros(3)
#             dmat: np.ndarray = np.zeros([3, 3])
#             dweight2: float = 0.0
#             # Get Scale Factor (referred in paper as "frequency")
#             domega: float = 2.0 * np.pi * dfreq

#             # Discrete wavelet transform (DWT)
#             # Lots of math here, but basically doing the summations shown in the paper
#             for idat in range(nstart, numdat):
#                 # Get dweight (an element of "local data number" viewed as "weights" in the paper)
#                 dz: float = domega * (timestamps[idat] - dtau)
#                 dweight: float = np.exp(-1 * decay_constant * dz ** 2)

#                 # get upper triangular matrix of the weights and vector
#                 # These are used later to calculate Neff, DWT, DWP, etc in the paper
#                 if dweight > 10 ** -9:
#                     cos_dz: float = np.cos(dz)
#                     sin_dz: float = np.sin(dz)
#                     dweight2 += dweight ** 2
#                     dvarw += dweight * magnitudes[idat] ** 2  # Used to get "weighted variation" later

#                     dmat[0, 0] += dweight
#                     dmat[0, 1] += dweight * cos_dz
#                     dmat[0, 2] += dweight * sin_dz
#                     dmat[1, 1] += dweight * cos_dz ** 2
#                     dmat[1, 2] += dweight * cos_dz * sin_dz
#                     dmat[2, 2] += dweight * sin_dz ** 2

#                     # parallel to the 3 trial functions (5-5, 6, 7)
#                     dvec[0] += dweight * magnitudes[idat]
#                     dvec[1] += dweight * magnitudes[idat] * cos_dz
#                     dvec[2] += dweight * magnitudes[idat] * sin_dz

#                 elif dz > 0:
#                     break
#                 else:
#                     nstart = idat + 1

#             # Get dneff ("effective number" for weighted projection)
#             if dweight2 > 0:
#                 # This is equation 5-4 in the paper
#                 dneff: float = (dmat[0, 0] ** 2) / dweight2
#             else:
#                 dneff = 0.0

#             # Get damp, dpower, dpowz
#             dcoef: List[int] = [0, 0, 0]

#             if dneff > 3:
#                 dvec = dvec / dmat[0, 0]
#                 # avoid for loops
#                 dmat[..., 1:] /= dmat[0, 0]

#                 if dmat[0, 0] > 0.005:
#                     dvarw = dvarw / dmat[0, 0]
#                 else:
#                     dvarw = 0.0

#                 # some initialize
#                 dmat[0, 0] = 1.0
#                 davew: float = dvec[0]
#                 dvarw = dvarw - (davew ** 2)  # "weighted variation" eq. 5-9

#                 if dvarw <= 0.0:
#                     dvarw = 10 ** -12

#                 # avoid for loops
#                 dmat[1, 0] = dmat[0, 1]
#                 dmat[2, 0] = dmat[0, 2]
#                 dmat[2, 1] = dmat[1, 2]

#                 if np.linalg.det(dmat) == 0:
#                     dmat = np.linalg.pinv(dmat)
#                     print("determinant is zero, using pseudo-inverse.")
#                 else:
#                     dmat = np.linalg.inv(dmat)

#                 # set dcoef and dpower
#                 dcoef = dmat.dot(dvec)  # y1, y2, and y3 from eq. 4-4, with 5-5, 6, 7
#                 dpower = np.dot(dcoef, dvec) - (davew ** 2)  # weighted model function eq. 5-10

#                 dpowz: float = (dneff - 3.0) * dpower / (2.0 * (dvarw - dpower))  # WWZ eq. 5-12
#                 damp = np.sqrt(dcoef[1] ** 2 + dcoef[2] ** 2)  # WWA eq. 5-14
#             else:
#                 dpowz = 0.0
#                 damp = 0.0

#             if dneff < (10 ** (-9)):
#                 dneff = 0.0

#             if damp < (10 ** (-9)):
#                 damp = 0.0

#             if dpowz < (10 ** (-9)):
#                 dpowz = 0.0

#             # Let's write everything out.
#             output[index] = [dtau, dfreq, dpowz, damp, dcoef[0], dneff]

#             index = index + 1

#         return output

#     # Check if parallel or not
#     if parallel:
#         output = np.array(Parallel(n_jobs=num_cores)(delayed(tau_loop)(dtau) for dtau in tau))
#     else:
#         output = np.empty([ntau, nfreq, 6])
#         for i, dtau in enumerate(tau):
#             output[i] = tau_loop(dtau)

#     # Format the output to be in len(tau) by len(freq) matrix for each value with correct labels

#     tau_mat: np.ndarray = output[:, :, 0].reshape([ntau, nfreq])
#     freq_mat: np.ndarray = output[:, :, 1].reshape([ntau, nfreq])
#     wwz_mat: np.ndarray = output[:, :, 2].reshape([ntau, nfreq])
#     amp_mat: np.ndarray = output[:, :, 3].reshape([ntau, nfreq])
#     dcoef_mat: np.ndarray = output[:, :, 4].reshape([ntau, nfreq])
#     dneff_mat: np.ndarray = output[:, :, 5].reshape([ntau, nfreq])

#     output = np.array([tau_mat, freq_mat, wwz_mat, amp_mat, dcoef_mat, dneff_mat])

#     # Finished Weighted Wavelet Z-transform and finish timer...
#     print(round(time.time() - process_starttime, 2), 'seconds has passed to complete Weighted Wavelet Z-transform \n')

#     return output



def wwt(timestamps: np.ndarray,
        magnitudes: np.ndarray,
        time_divisions: int,
        freq_params: list,
        decay_constant: float,
        method: str = 'linear',
        parallel: bool = True) -> np.ndarray:
    """
    The code is based on G. Foster's FORTRAN
    code as well as eaydin's python 2.7 code. The code is updated to use numpy methods and allow for float value tau.
    It returns an array with matrix of new evenly spaced timestamps, frequencies, wwz-power, amplitude, coefficients,
    and effective number. Specific equations can be found on Grant Foster's "WAVELETS FOR PERIOD ANALYSIS OF UNEVENLY
    SAMPLED TIME SERIES". Some of the equations are labeled in the code with corresponding numbers.

    :param timestamps: An array with corresponding times for the magnitude (payload).
    :param magnitudes: An array with payload values
    :param time_divisions: number of divisions for the new timestamps
    :param freq_params: A list containing parameters for making frequency bands to analyze over with given 'method'
            'linear' -> [freq_low, freq_high, freq_step, override]
            'octave' -> [freq_tg, freq_low, freq_high, band_order, log_scale_base, override]
    :param decay_constant: decay constant for the Morlet wavelet (should be <0.02) eq. 1-2
            c = 1/(2w), the wavelet decays significantly after a single cycle of 2 * pi / w
    :param method: determines method of creating freq ('linear', 'octave') default 'linear'
    :param parallel: boolean indicate to use parallel processing or not
    :return: Tau, Freq, WWZ, AMP, COEF, NEFF in a numpy array
    """
#     returned=[]#del
    # Starting Weighted Wavelet Z-transform and start timer...
    print("*** Starting Weighted Wavelet Z-transform ***\n")
    process_starttime: float = time.time()

    # Get taus to compute WWZ (referred in paper as "time shift(s)")
    tau: np.ndarray = make_tau(timestamps, time_divisions)
    ntau: int = len(tau)

    # Calculate pseudo sample rate and largest time window to check for requirements
    freq_pseudo_sr = 1 / np.median(np.diff(timestamps))  # 1 / median period

    # noinspection PyArgumentList
    largest_tau_window = tau[1] - tau[0]
    print('Pseudo sample frequency (median) is ', np.round(freq_pseudo_sr, 3))
    print('largest tau window is ', np.round(largest_tau_window, 3))

    # Frequencies to compute WWZ
    if method == 'linear':
        freq: np.ndarray = freq_params # edited - directly feed in frequency steps
        # freq: np.ndarray = make_freq(freq_low=freq_params[0],
        #                              freq_high=freq_params[1],
        #                              freq_steps=freq_params[2])
        nfreq: int = len(freq)

    elif method == 'octave':
        freq = make_octave_freq(freq_target=freq_params[0],
                                freq_low=freq_params[1],
                                freq_high=freq_params[2],
                                band_order=freq_params[3],
                                log_scale_base=freq_params[4],
                                freq_pseudo_sr=freq_pseudo_sr,
                                largest_tau_window=largest_tau_window,
                                override=freq_params[5])
        nfreq = len(freq)

    # Get number of data from timestamps
    numdat: int = len(timestamps)

    # Get number of CPU cores on current device (used for parallel)
    num_cores = multiprocessing.cpu_count()

    # WWT Starts Here
    dvec = torch.zeros((ntau, nfreq, 3))
    dmat = torch.zeros((ntau, nfreq, 3, 3))
    dweight2 = torch.zeros((ntau, nfreq))
    domega =  torch.tensor(2.0 * np.pi * freq)
    dz = torch.einsum('ij,k->ikj',torch.tensor(timestamps - tau[:,np.newaxis]), domega)
    dweight = torch.exp(-1 * decay_constant * dz ** 2)

    dweight_mask = dweight <= 10 ** -9
    dweight[dweight_mask] = 0 
    cos_dz = torch.cos(dz)
    sin_dz = torch.sin(dz)
    cos_dz[dweight_mask] = 0
    sin_dz[dweight_mask] = 0
                
            
    dweight2 = torch.sum(dweight ** 2, axis=2)
    dvarw = torch.sum(dweight * magnitudes ** 2, axis=2)
    dmat[:, :, 0, 0] = torch.sum(dweight, axis=2)
    dmat[:, :, 0, 1] = torch.sum(dweight * cos_dz, axis=2)
    dmat[:, :, 0, 2] = torch.sum(dweight * sin_dz, axis=2)
    dmat[:, :, 1, 1] = torch.sum(dweight * cos_dz ** 2, axis=2)
    dmat[:, :, 1, 2] = torch.sum(dweight * cos_dz * sin_dz, axis=2)
    dmat[:, :, 2, 2] = torch.sum(dweight * sin_dz ** 2, axis=2)

    dvec[:, :, 0] = torch.sum(dweight * magnitudes, axis=2)
    dvec[:, :, 1] = torch.sum(dweight * magnitudes * cos_dz, axis=2)
    dvec[:, :, 2] = torch.sum(dweight * magnitudes * sin_dz, axis=2)
            
    dneff = torch.zeros((ntau,nfreq))
    dneff[dweight2 > 0] = (dmat[:, :, 0, 0][dweight2 > 0] ** 2) / dweight2[dweight2 > 0].float()

    dcoef = torch.zeros((ntau, nfreq, 3))
    dpower = torch.zeros((ntau, nfreq))
    dpowz = torch.zeros((ntau, nfreq))
    damp = torch.zeros((ntau, nfreq))
        
        # boolean mask instead of the if statement
    dneff_mask = dneff > 3 
#     dneff_mask3d = dneff_mask.unsqueeze(-1).expand(dvec.size())
#     dneff_mask4d = dneff_mask3d.unsqueeze(-1).expand(dmat.size()) 
    # https://stackoverflow.com/questions/61956893/how-to-mask-a-3d-tensor-with-2d-mask-and-keep-the-dimensions-of-original-vector
    dvec[] = (dvec[dneff_mask3d].T / dmat[dneff_mask4d][:,0,0]).T
    dmat[dneff_mask, :, 1:] = (dmat[dneff_mask, :, 1:].T / dmat[dneff_mask, 0, 0]).T
    dmat_mask = dmat[:, :, 0, 0] > 0.005 # S[0] > 0.0 in vartools?
    dvarw[dmat_mask] = dvarw[dmat_mask] / dmat[dmat_mask, 0, 0]
    dvarw[~dmat_mask] = 0.0
    dmat[dneff_mask, 0, 0] = 1.0
    davew = dvec[:, :, 0]
    dvarw = dvarw - (davew ** 2)
    dvarw[dvarw <= 0.0] = 10 ** -12
    dmat[dneff_mask, 1, 0] = dmat[dneff_mask, 0, 1] # could avoid using the mask perhaps
    dmat[dneff_mask, 2, 0] = dmat[dneff_mask, 0, 2]
    dmat[dneff_mask, 2, 1] = dmat[dneff_mask, 1, 2]
    dmat[dneff_mask, :, :] = dmat[dneff_mask, :, :].pinverse()
    #dcoef[dneff_mask, :] = dmat[dneff_mask, :, :].dot(dvec[dneff_mask,:])  # y1, y2, and y3 from eq. 4-4, with 5-5, 6, 7
    #ValueError: shapes (400,3,3) and (400,3) not aligned: 3 (dim 2) != 400 (dim 0)
    dcoef[dneff_mask, :] = torch.einsum('ijk,ik->ij', dmat[dneff_mask, :, :], dvec[dneff_mask,:])
    #dpower = np.dot(dcoef[dneff_mask, :], dvec[dneff_mask, :]) - (davew[dneff_mask] ** 2)  # weighted model function eq. 5-10
    #ValueError: shapes (400,3) and (400,3) not aligned: 3 (dim 1) != 400 (dim 0)
    dpower[dneff_mask] = torch.einsum('ij,ij->i',dcoef[dneff_mask, :], dvec[dneff_mask, :]) - (davew[dneff_mask] ** 2)
#         print((dpowz.shape, dneff_mask.shape, dneff.shape, dpower.shape, dvarw.shape))
    dpowz[dneff_mask] = (dneff[dneff_mask] - 3.0) * dpower[dneff_mask] / (2.0 * (dvarw[dneff_mask] - dpower[dneff_mask]))  # WWZ eq. 5-12
    damp[dneff_mask] = torch.sqrt(dcoef[dneff_mask, 1] ** 2 + dcoef[dneff_mask, 2] ** 2)  # WWA eq. 5-14

    dneff[dneff < 10e-9] = 0.0
    damp[damp < 10e-9] = 0.0
    dpowz[dpowz < 10e-9] = 0.0
        

        
#     output = np.empty([6, ntau, nfreq])

#     # Format the output to be in len(tau) by len(freq) matrix for each value with correct labels
# #     print((output[:, :, 0].shape, output.shape))
#     tau_mat: np.ndarray = output[:, :, 0].reshape([ntau, nfreq])
#     freq_mat: np.ndarray = output[:, :, 1].reshape([ntau, nfreq])
#     wwz_mat: np.ndarray = output[:, :, 2].reshape([ntau, nfreq])
#     amp_mat: np.ndarray = output[:, :, 3].reshape([ntau, nfreq])
#     dcoef_mat: np.ndarray = output[:, :, 4].reshape([ntau, nfreq])
#     dneff_mat: np.ndarray = output[:, :, 5].reshape([ntau, nfreq])

#     for xxx in [np.tile(tau,(len(freq),1)).T, np.tile(freq, (len(tau),1)), dpowz, damp, dcoef[:, 0], dneff]:
#         print(xxx.shape)

    output = torch.tensor([torch.tile(tau,(len(freq),1)).T, torch.tile(freq, (len(tau),1)), dpowz, damp, dcoef[:,:, 0], dneff])

    # Finished Weighted Wavelet Z-transform and finish timer...
    print(round(time.time() - process_starttime, 2), 'seconds has passed to complete Weighted Wavelet Z-transform \n')

    return output.numpy()