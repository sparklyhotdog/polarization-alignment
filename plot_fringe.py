import matplotlib.pyplot as plt
import numpy as np
from measurements import measure_for_plot
from scipy.spatial.transform import Rotation as r


def plot(ret_angles=None, num_points=10, title=None, filepath=None, verbose=False, expected_count_rates=None):
    """Plots the measured power (mW) while spanning the 4th wave plate rotation angle in the polarization analyzer for
    each of the H, V, D, A states.

    ret_angles is a array of 6 retardances (degrees) for the waveplates. If expected_count_rates
    (length 4 array, for H, V, D, A) are provided, the expected sinusoids are plotted with dotted lines."""

    if ret_angles is None:
        ret_angles = np.zeros(6)

    bases = ['H', 'V', 'D', 'A', 'R', 'L']

    x = np.linspace(0, 360, num_points)
    x_radians = x * np.pi / 180

    readings = measure_for_plot(ret_angles, verbose=verbose)

    fig, ax = plt.subplots()
    for i in range(6):
        ax.plot(x, readings[i], label=bases[i])

    if expected_count_rates is not None:
        ax.plot(x, expected_count_rates[0] / 2 * (1 + np.cos(x_radians)), linestyle=':', color='tab:blue', alpha=0.5)
        ax.plot(x, expected_count_rates[1] / 2 * (1 - np.cos(x_radians)), linestyle=':', color='tab:orange', alpha=0.5)
        ax.plot(x, expected_count_rates[2] / 2 * (1 + np.sin(x_radians)), linestyle=':', color='tab:green', alpha=0.5)
        ax.plot(x, expected_count_rates[3] / 2 * (1 - np.sin(x_radians)), linestyle=':', color='tab:red', alpha=0.5)

    ax.set(xlabel='Theta for Waveplate 4 (deg)', ylabel='Power (mW)', title=title)
    legend = plt.legend()
    ax.grid()
    if filepath is not None:
        plt.savefig(filepath)

    plt.show()


def plot2(ret_angles=None, num_points=10, title=None, filepath=None, verbose=False, expected_count_rates=None):
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
    x_radians = x * np.pi / 180

    readings0 = measure_for_plot(ret_angles[0], verbose=verbose)
    readings1 = measure_for_plot(ret_angles[1], verbose=verbose)

    ax0 = plt.subplot(211)

    for i in range(6):
        plt.plot(x, readings0[i], label=bases[i])

    if expected_count_rates is not None:
        plt.plot(x, expected_count_rates[0] / 2 * (1 + np.cos(x_radians)), linestyle=':', color='tab:blue', alpha=0.5)
        plt.plot(x, expected_count_rates[1] / 2 * (1 - np.cos(x_radians)), linestyle=':', color='tab:orange', alpha=0.5)
        plt.plot(x, expected_count_rates[2] / 2 * (1 + np.sin(x_radians)), linestyle=':', color='tab:green', alpha=0.5)
        plt.plot(x, expected_count_rates[3] / 2 * (1 - np.sin(x_radians)), linestyle=':', color='tab:red', alpha=0.5)

    plt.ylabel('Power (mW)')
    plt.title(title)
    plt.legend()
    plt.grid()

    ax1 = plt.subplot(212, sharex=ax0, sharey=ax0)

    for i in range(6):
        plt.plot(x, readings1[i], label=bases[i])

    if expected_count_rates is not None:
        plt.plot(x, expected_count_rates[0] / 2 * (1 + np.cos(x_radians)), linestyle=':', color='tab:blue', alpha=0.5)
        plt.plot(x, expected_count_rates[1] / 2 * (1 - np.cos(x_radians)), linestyle=':', color='tab:orange', alpha=0.5)
        plt.plot(x, expected_count_rates[2] / 2 * (1 + np.sin(x_radians)), linestyle=':', color='tab:green', alpha=0.5)
        plt.plot(x, expected_count_rates[3] / 2 * (1 - np.sin(x_radians)), linestyle=':', color='tab:red', alpha=0.5)

    plt.xlabel('Theta Offset for Waveplate 4 (deg)')
    plt.ylabel('Power (mW)')
    plt.legend()
    plt.grid()
    if filepath is not None:
        plt.savefig(filepath)
    plt.show()




if __name__ == "__main__":
    P = np.zeros(6)

    # Set the angles for the first 3 waveplates to emulate inverse T.
    # The xyx euler angles of T are chosen from the txt output file from the least squares fitting
    leastsquares_output = np.loadtxt('data/leastsquares_output.txt')
    avg = (leastsquares_output[5] + leastsquares_output[6])/2
    count_rates = [leastsquares_output[5], avg, leastsquares_output[6], avg]
    thetas = leastsquares_output[0:3]
    # P[0:3] = thetas
    P[0:3] = [360 - thetas[2], 360 - thetas[1], 360 - thetas[0]]

    # Set the offsets to undo F (taken from the NuCrypt config info)
    f_theta = leastsquares_output[3]
    f_phi = leastsquares_output[4]
    f = np.asarray([np.cos(f_theta) * np.sin(f_phi), np.sin(f_theta) * np.sin(f_phi), np.cos(f_phi)])

    P[3] = 360 - np.arccos(f[0])*180/np.pi
    P[4] = np.arccos(f[2]/np.sqrt(f[1]**2 + f[2]**2))*180/np.pi
    if f[1] > 0:
        P[4] = 360 - P[4]
    # P[3:6] = 360*np.ones(3) - np.asarray([117.972, 14.7918, 150.632])
    # P[3:6] = 360*np.ones(3) - P[3:6]
    P = np.zeros(6)

    print("Offsets for the waveplates: ", P)
    plot(P, title=str(P), filepath='plots/jun30_nocompensation.png', verbose=True)

    # plot2(np.zeros((2, 6)), expected_count_rates=[.8, .8, .8, .8])
