"""
This module provides functions for plotting the outcomes of wwz.py. It focuses on making proper grids for pcolormesh.
"""

import matplotlib.ticker as ticker
import matplotlib.axes as axes
import numpy as np
import matplotlib.ticker as ticker # edited - needed for log plotter
from matplotlib.ticker import FormatStrFormatter # edited - needed for log plotter



def make_linear_freq_plot_grid(freq_mat: np.ndarray) -> np.ndarray:
    """
    Used for linear method.
    Takes the FREQ output from wwz.py and creates a grid for pcolormesh.
    :param freq_mat: FREQ output from wwz.py
    :return: freq_grid: np.ndarray with the boundaries for the FREQ output
    """

    # Get the array of center frequencies from the freq_mat
    freq_centers = freq_mat[0, :]

    # Get the freq_steps by subtracting the first two freq_centers
    freq_step = freq_centers[1] - freq_centers[0]

    # Subtract half of the freq_step from the freq_centers to get lower bound
    freq_lows = freq_centers - freq_step / 2

    # Append the high frequency bound to get all the boundaries
    freq_highest = freq_centers.max() + freq_step / 2
    freq_bounds = np.append(freq_lows, freq_highest)

    # Tile the freq_bounds to create a grid
    freq_grid = np.tile(freq_bounds, (freq_mat.shape[0] + 1, 1))

    return freq_grid


def make_octave_freq_plot_grid(freq_mat: np.ndarray, band_order: float, log_scale_base: float) -> np.ndarray:
    """
    Used for octave method.
    Takes the FREQ output from wwz.py and creates a grid for pcolormesh
    :param freq_mat: FREQ output from wwz.py
    :param band_order: octave band order (N => 1) *Recommend N = (1, 3, 6, 12, 24,...)
    :param log_scale_base: logarithmic scale base to create the octaves
    :return:req_grid: np.ndarray with the boundaries for the FREQ output
    """

    # Get the array of the center frequencies from the freq_mat
    freq_centers = freq_mat[0, :]

    # Convert the center frequencies to low frequencies
    freq_lows = freq_centers / log_scale_base**(1 / (2 * band_order))

    # Append the high frequency at the end to get all the boundaries
    freq_highest = freq_centers.max() * log_scale_base**(1 / (2 * band_order))
    freq_bounds = np.append(freq_lows, freq_highest)

    # Tile the freq_bounds to create a grid
    freq_grid = np.tile(freq_bounds, (freq_mat.shape[0] + 1, 1))

    return freq_grid


def make_tau_plot_grid(tau_mat: np.ndarray) -> np.ndarray:
    """
    Used for both octave and linear.
    Takes the TAU output from wwz.py and creates a grid for pcolormesh
    :param tau_mat: TAU output from wwz.py
    :return:
    """

    # Get the tau values from tau_mat
    taus = tau_mat[:, 0]

    # Append one tau value for edge limit by adding the step to the largest tau
    taus = np.append(taus, taus[-1] + taus[1] - taus[0])

    # Tile the taus with an additional column to create grid that matches freq_grid
    tau_grid = np.tile(taus, (tau_mat.shape[1] + 1, 1)).transpose()

    return tau_grid


def linear_plotter(ax: axes, TAU: np.ndarray, FREQ: np.ndarray, DATA: np.ndarray):
    """
    Creates a plot for the 'linear' method.
    You can add titles after calling the plotter.
    :param ax: axis from a matplotlib.pyplot.subplots() to plot the data
    :param TAU: TAU output of the wwz.py (the time shifts)
    :param FREQ: FREQ output of the wwz.py (the frequencies)
    :param DATA: Desired data to be plotted
    :return:
    """

    # Create grid for pcolormesh boundaries
    tau_grid = make_tau_plot_grid(TAU)
    freq_grid = make_linear_freq_plot_grid(FREQ)

    # Plot using subplots
    im = ax.pcolormesh(tau_grid, freq_grid, DATA)

    # Add color bar and fix y_ticks
    ax.figure.colorbar(im, ax=ax)
    ax.set_yticks(FREQ[0, :])
    ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

def masked_plotter(ax: axes, TAU: np.ndarray, FREQ: np.ndarray, DATA: np.ndarray, clip=None):  # edited - new function
    """
    Edited linear plotter function. The color bar has a logarithmic. Clipping of
    values is implemented. Zero values are masked to allow logarithmic axis scale

    You can add titles after calling the plotter.
    :param ax: axis from a matplotlib.pyplot.subplots() to plot the data
    :param TAU: TAU output of the wwz.py (the time shifts)
    :param FREQ: FREQ output of the wwz.py (the frequencies)
    :param DATA: Desired data to be plotted
    :param clip: expects [minimum, maximum] list of values fed to numpy.clip
    :return:
    """
    #mask zero values, required to allow log scale axis
    DATA_masked = np.ma.masked_where(DATA == 0, DATA)
    data_mean = DATA_masked.mean()
    data_std = DATA_masked.std()
    if clip:
        DATA_clipped = np.clip(DATA_masked, clip[0], clip[1])
    else:
        DATA_clipped = DATA_masked

    # Create grid for pcolormesh boundaries
    tau_grid = make_tau_plot_grid(TAU)
    freq_grid = make_linear_freq_plot_grid(FREQ)

    # Plot using subplots
    im = ax.pcolormesh(tau_grid, freq_grid, DATA_clipped, norm=colors.LogNorm(vmin=DATA_clipped.min(), vmax=DATA_clipped.max()))

    # Add color bar and fix y_ticks
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.1)
    ax.set_yticks(FREQ[0, :])
    ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    cbar.ax.set_ylabel('wavelet power', rotation=90)

def log_plotter(ax: axes, TAU: np.ndarray, FREQ: np.ndarray, DATA: np.ndarray, clip=None): # edited - new function
    """
    Improves the output of masked_plotter, changing axis scale to logarithmic,
    it is assumed that data is timed in days, y axis is logarithmically spaced
    in terms of frequency but a secondary y axis also shows corresponding period values
    """

    masked_plotter(ax, TAU, FREQ, DATA, clip)
    ax.set_yscale('log')
    def forward(x):
        return 1 / x
    def inverse(x):
        return 1 / x
    secax = ax.secondary_yaxis('right')#, functions=(forward, inverse))
    secax.set_ylabel('frequency ($days^{-1}$)', labelpad=0)
    ax.tick_params(axis='y', which='minor')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: 1/y))
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda y, _: np.round(1/y, decimals=2)))
    ax.set_ylabel('period (days)')
    ax.set_xlabel('Date (JD)')
    plt.tight_layout()


def octave_plotter(ax: axes, TAU: np.ndarray, FREQ: np.ndarray, DATA: np.ndarray,
                   band_order: float, log_scale_base: float, log_y_scale: bool = True):
    """
    Creates a plot for the 'linear' method.
    You can add titles after calling the plotter.
    :param ax: axis from a matplotlib.pyplot.subplots() to plot the data
    :param TAU: TAU output of the wwz.py (the time shifts)
    :param FREQ: FREQ output of the wwz.py (the frequencies)
    :param DATA: Desired data to be plotted
    :param band_order: octave band order (N => 1) *Recommend N = (1, 3, 6, 12, 24,...)
    :param log_scale_base: logarithmic scale base to create the octaves
    :param log_y_scale: determines if the plot y_scale should be 'log' or not.
    :return:
    """
    # Create grid for pcolormesh boundaries
    tau_grid = make_tau_plot_grid(TAU)
    freq_grid = make_octave_freq_plot_grid(FREQ, band_order, log_scale_base)

    # Plot using subplots
    im = ax.pcolormesh(tau_grid, freq_grid, DATA)

    # Add color bar, fix y_scale, and fix y_ticks
    ax.figure.colorbar(im, ax=ax)
    if log_y_scale is True:
        ax.set_yscale('log', base=log_scale_base)
    ax.set_yticks(FREQ[0, :])
    ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
