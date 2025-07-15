import matplotlib.pyplot as plt
import numpy as np
from measurements import measure_for_plot
from scipy.spatial.transform import Rotation as r


def plot(ret_angles=None, num_points=10, title=None, filepath=None, verbose=False):
    """Plots the measured power (mW) while spanning the 4th wave plate rotation angle in the polarization analyzer for
    each of the H, V, D, A states.

    ret_angles is an array of 6 retardances (degrees) for the waveplates. """

    if ret_angles is None:
        ret_angles = np.zeros(6)

    bases = ['H', 'V', 'D', 'A', 'R', 'L']

    x = np.linspace(0, 360, num_points)

    readings = measure_for_plot(ret_angles, verbose=verbose, num_points=num_points)

    fig, ax = plt.subplots()
    for i in range(6):
        ax.plot(x, readings[i], label=bases[i])

    plt.xlabel('Theta Offset for Waveplate 4 (deg)', fontsize='x-large')
    plt.ylabel('Power (mW)', fontsize='x-large')
    plt.title(title, fontsize='xx-large')
    legend = plt.legend()
    ax.grid()
    if filepath is not None:
        plt.savefig(filepath)

    plt.show()


def plot2(ret_angles=None, num_points=10, title=None, filepath=None, verbose=False):
    """Creates 2 plots for the measured power (mW) while spanning the 4th wave plate rotation angle for each of the
    H, V, D, A states.

    ret_angles is an 2x6 array of retardances (degrees) for the waveplates.
    ret_angles[0] is the first set of angles for the first plot; ret_angles[1] is for the second plot.
    If expected_count_rates (length 4 array, for H, V, D, A) are provided, the expected sinusoids are plotted with
    dotted lines."""

    if ret_angles is None:
        ret_angles = np.zeros((2, 6))

    bases = ['H', 'V', 'D', 'A', 'R', 'L']
    x = np.linspace(0, 360, num_points)

    readings0 = measure_for_plot(ret_angles[0], verbose=verbose)
    readings1 = measure_for_plot(ret_angles[1], verbose=verbose)

    ax0 = plt.subplot(211)

    for i in range(6):
        plt.plot(x, readings0[i], label=bases[i])

    plt.ylabel('Power (mW)', fontsize='x-large')
    plt.title(title, fontsize='xx-large')
    plt.legend()
    plt.grid()

    ax1 = plt.subplot(212, sharex=ax0, sharey=ax0)

    for i in range(6):
        plt.plot(x, readings1[i], label=bases[i])

    plt.xlabel('Theta Offset for Waveplate 4 (deg)', fontsize='x-large')
    plt.ylabel('Power (mW)', fontsize='x-large')
    plt.legend()
    plt.grid()
    if filepath is not None:
        plt.savefig(filepath)
    plt.show()


def plot_expected(filepath=None):
    """"""

    bases = ['H', 'V', 'D', 'A', 'R', 'L']
    num_points = 100
    x = np.linspace(0, 360, num_points)
    theta = x * np.pi / 180
    counts = 1

    readings = [
        0.5 * counts * (1 + np.cos(theta)),
        0.5 * counts * (1 - np.cos(theta)),
        0.5 * counts * np.ones(num_points),
        0.5 * counts * np.ones(num_points),
        0.5 * counts * (1 + np.sin(theta)),
        0.5 * counts * (1 - np.sin(theta)),
    ]

    fig, ax = plt.subplots()
    for i in range(6):
        ax.plot(x, readings[i], label=bases[i])

    ax.set_ylabel(ylabel='Fraction of max power', fontsize='x-large')
    ax.set_title(label='Expected', fontsize='xx-large')
    ax.set_xlabel(xlabel='Theta for Waveplate 4 (deg)', fontsize='x-large')
    legend = plt.legend()
    ax.grid()
    if filepath is not None:
        plt.savefig(filepath)

    plt.show()


if __name__ == "__main__":
    # P = np.zeros(6)
    #
    # print("Offsets for the waveplates: ", P)
    # plot(P, title=str(P), filepath='plots/jun30_nocompensation.png', verbose=True)
    plot_expected(filepath='plots/expected.png')

