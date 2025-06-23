import numpy as np
from client import Client
from scipy.spatial.transform import Rotation as r
import yaml


def measure(rotations, yaml_fn, verbose=False, datapath=None):
    """Given a list of rotations in terms of its angles (nx3 matrix), returns the powermeter measurements corresponding
    to the H and D states. The last three wave plates will be set to 0"""
    num_rotations = rotations.shape[0]
    rotations = rotations % 360     # the polarization analyzer accepts angles in degrees with range [0, 420]

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
    actual_angles = np.empty((num_rotations, 6))
    for i in range(num_rotations):
        # set the plate angles in the polarization analyzer
        for j in range(3):
            msg = 'retard ' + str(j) + ' ' + str(rotations[i][j])
            resp = pol_analyzer._send(msg)

            if verbose:
                print(msg)
                print(resp)

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

        if datapath is not None:
            np.savetxt(datapath, counts)

    return counts, actual_angles


if __name__ == "__main__":

    rotation_list = r.as_euler(r.random(1), "xyx")
    # print(rotation_list)

    measurements, angles = measure(rotation_list, yaml_fn='serverinfo.yaml', verbose=True)

    print(measurements)
    print(angles)
