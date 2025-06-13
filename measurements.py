import numpy as np
from client import Client
from scipy.spatial.transform import Rotation as r
import yaml


def generate_eulerangles(rotations=None, rounding_digits=2):
    """theta1, theta3: [0, 360]
    theta2: [0, 180]"""
    if rotations is None:
        rots = r.random(16).as_euler("xyx", degrees=True)
    else:
        rots = rotations.as_euler("xyx", degrees=True)
    # scipy rotation library returns its euler angles with ranges [-180, 180], [0, 180], [-180, 180]
    # but the polarization analyzer only accepts pos angles

    for rot in rots:
        if rot[0] < 0:
            rot[0] += 180
        if rot[2] < 0:
            rot[2] += 180

    rots = np.round(rots, rounding_digits)

    return rots


def measure(num_rotations, rotations, yaml_fn, verbose=False, datapath=None):
    """Takes powermeter measurements given a list of rotations (a nx3 matrix of euler angles)"""
    y_fn = open(yaml_fn)
    dicty = yaml.load(y_fn, Loader=yaml.SafeLoader)
    y_fn.close()

    pol_analyzer = Client(dicty['PA1000']['host'], dicty['PA1000']['port'])
    synthesizer = Client(dicty['PSY201']['host'], dicty['PSY201']['port'])
    power_meter = Client(dicty['PM400']['host'], dicty['PM400']['port'])

    # We will only use the first three wave plates, so we set the last 3 to be 0
    pol_analyzer._send('retard 3 0')
    pol_analyzer._send('retard 4 0')
    pol_analyzer._send('retard 5 0')

    counts = np.empty(2 * num_rotations)
    for i in range(num_rotations):
        # set the plate angles in the polarization analyzer
        for j in range(3):
            msg = 'retard ' + str(j) + ' ' + str(rotations[i][j])
            resp = pol_analyzer._send(msg)
            if verbose:
                print(msg)
                print(resp)

        # make measurements (in mW)
        synthesizer._send('h')
        counts[2 * i] = 1000 * float(power_meter._send('pow?'))
        synthesizer._send('d')
        counts[2 * i + 1] = 1000 * float(power_meter._send('pow?'))

        if datapath is not None:
            np.savetxt(datapath, counts)

    return counts


if __name__ == "__main__":

    n = 16
    rotation_list = generate_eulerangles()
    # print(rotation_list)

    measurements = measure(n, rotation_list, yaml_fn='serverinfo.yaml', verbose=True, datapath='data/data.txt')

    print(measurements)
