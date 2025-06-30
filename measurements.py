import numpy as np
from client import Client
from scipy.spatial.transform import Rotation as r
import yaml
import time


def measure_HD(rotations, verbose=False, datapath=None, rotpath=None):
    """Given a list of rotations in terms of its angles (nx3 matrix), returns the powermeter measurements corresponding
    to the H and D states, and the angles set during the measurements. The last three wave plates will be set to 0"""
    num_rotations = rotations.shape[0]
    rotations = rotations % 360     # the polarization analyzer accepts angles in degrees with range [0, 420]
    rotations = np.round(rotations, 10)     # the PA gets upset if there are too many decimals

    y_fn = open('serverinfo.yaml')
    dicty = yaml.load(y_fn, Loader=yaml.SafeLoader)
    y_fn.close()

    pol_analyzer = Client(dicty['PA1000']['host'], dicty['PA1000']['port'])
    synthesizer = Client(dicty['PSY201']['host'], dicty['PSY201']['port'])
    power_meter = Client(dicty['PM400']['host'], dicty['PM400']['port'])

    # We will only use the first three wave plates to set the rotations, so we set the last 3 to be 0
    pol_analyzer._send('zeroall')

    counts = np.empty(2 * num_rotations)
    actual_angles = np.empty((num_rotations, 6))
    for i in range(num_rotations):
        # set the plate angles in the polarization analyzer
        for j in range(3):
            msg = 'retard ' + str(j) + ' ' + str(rotations[i][j])
            resp = pol_analyzer._send(msg)

            if verbose:
                print(msg)
                print(resp)

        # record the angles given back by the polarization analyzer
        resp = resp.strip("[]")
        arr = resp.split(", ")
        for j in range(6):
            arr[j] = float(arr[j].strip("'"))
        actual_angles[i] = arr

        # make measurements (in mW)
        synthesizer._send('h')
        counts[2 * i] = 1000 * float(power_meter._send('pow?'))
        synthesizer._send('d')
        counts[2 * i + 1] = 1000 * float(power_meter._send('pow?'))

        if verbose:
            print("Measurements (mW):", counts[2 * i], counts[2 * i + 1])

        if datapath is not None:
            np.savetxt(datapath, counts)
        if rotpath is not None:
            np.savetxt(rotpath, actual_angles)

    return counts, actual_angles


def measure_for_plot(ret_angles, num_points=10, verbose=False):
    """Returns measurements (mW) (4 x num_points array) from spanning the fourth waveplate rotation angle,
    given a list of retardance angles (in degrees) to set the wave plates (should be an array of length 6). """

    y_fn = open('serverinfo.yaml')
    dicty = yaml.load(y_fn, Loader=yaml.SafeLoader)
    y_fn.close()

    analyzer = Client(dicty['PA1000']['host'], dicty['PA1000']['port'])
    synth = Client(dicty['PSY201']['host'], dicty['PSY201']['port'])
    pow_meter = Client(dicty['PM400']['host'], dicty['PM400']['port'])

    for i in range(6):
        msg = 'retard ' + str(i) + ' ' + str(round(ret_angles[i], 10))
        resp = analyzer._send(msg)

        if verbose:
            print(msg)
            print(resp)

    # Start taking measurements
    bases = ['h', 'v', 'd', 'a', 'r', 'l']
    readings = np.zeros(shape=(6, num_points))

    x = np.linspace(0, 360, num_points)
    for i in range(6):
        synth._send(bases[i])  # Set polarization synthesizer to H, V, D, A

        for j in range(num_points):
            # Change the angle of the fourth plate
            msg = 'retard 3 ' + str(round((ret_angles[3] + round(x[j], 5)) % 360, 10))
            resp = analyzer._send(msg)

            readings[i][j] = (1000 * float(pow_meter._send('pow?')))  # Record the power measurement

            if verbose:
                print(msg)
                print(resp)

    analyzer.disconnect()
    synth.disconnect()
    pow_meter.disconnect()

    return readings


if __name__ == "__main__":

    rotation_list = r.as_euler(r.random(16), "xyx", degrees=True)

    measurements, angles = measure_HD(rotation_list, verbose=True)

    print(measurements)
    # print(angles)
