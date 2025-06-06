import numpy as np
from client import Client
from scipy.spatial.transform import Rotation as r


def generate_random_rotations(num_rotations=16, rounding_digits=2):
    """theta1, theta3: [0, 360]
    theta2: [0, 180]"""
    rots = r.random(num_rotations).as_euler("xyx", degrees=True)
    # scipy rotation library returns its euler angles with ranges [-180, 180], [0, 180], [-180, 180]
    # but the polarization analyzer only accepts pos angles

    for rot in rots:
        if rot[0] < 0:
            rot[0] += 180
        if rot[2] < 0:
            rot[2] += 180

    rots = np.round(rots, rounding_digits)

    return rots


def measure(num_rotations, rotation_list, verbose=False):
    HOST = '169.254.142.114'
    PA_port = 56000
    PSY_port = 56010
    PM_port = 56001

    pol_analyzer = Client(HOST, PA_port)
    synthesizer = Client(HOST, PSY_port)
    power_meter = Client(HOST, PM_port)

    # We will only use the first three wave plates, so we set the last 3 to be 0
    pol_analyzer._send('retard 3 0')
    pol_analyzer._send('retard 4 0')
    pol_analyzer._send('retard 5 0')

    counts = np.empty(2 * num_rotations)
    for i in range(num_rotations):
        # set the plate angles in the polarization analyzer
        for j in range(3):
            msg = 'retard ' + str(j) + ' ' + str(rotation_list[i][j])
            resp = pol_analyzer._send(msg)
            if verbose:
                print(msg)
                print(resp)

        # make measurements (in mW)
        synthesizer._send('h')
        counts[2 * i] = 1000 * float(power_meter._send('pow?'))
        synthesizer._send('d')
        counts[2 * i + 1] = 1000 * float(power_meter._send('pow?'))

    return counts


if __name__ == "__main__":

    n = 16
    rotation_list = generate_random_rotations(num_rotations=n)
    # print(rotation_list)

    measurements = measure(n, rotation_list)

    print(measurements)
