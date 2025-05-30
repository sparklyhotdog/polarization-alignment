import numpy as np
from scipy.spatial.transform import Rotation as r
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Generate an arbitrary rotation matrix (3x3, not a Mueller matrix)
    T = r.as_matrix(r.from_euler("xyz", [2*np.pi*random.random(), 2*np.pi*random.random(), 2*np.pi*random.random()]))
    N_H = random.randrange(500, 1500)
    N_D = random.randrange(500, 1500)

    # Pick >= 3 rotations for the PA to emulate
    R1 = r.as_matrix(r.from_rotvec([1, 0, 0]))
    R2 = r.as_matrix(r.from_rotvec([0, 1, 0]))
    R3 = r.as_matrix(r.from_rotvec([0, 0, 1]))
    rotations = [R1, R2, R3]
    n = len(rotations)

    C_H = np.asmatrix(np.empty((n, 1)))
    C_D = np.asmatrix(np.empty((n, 1)))
    P = np.asmatrix(np.empty((n, 3)))
    for i in range(n):
        # Calculate the photon counts for both the H and D states. This would be measured.
        C_H[i][0] = 0.5 * N_H * (1 + np.asmatrix([1, 0, 0]) * rotations[i] * T * np.asmatrix(np.asarray([[1], [0], [0]])))
        C_D[i][0] = 0.5 * N_D * (1 + np.asmatrix([1, 0, 0]) * rotations[i] * T * np.asmatrix(np.asarray([[0], [1], [0]])))

        # Also, construct our P matrix for all rotations. The ith row of P contains the first row of rotation matrix i.
        # We only care about the first row of each rotation, since we are multiplying it by the H polarizer matrix.
        P[i] = rotations[i][0]

    p_inverse = np.linalg.pinv(P)

    # Now, we want to solve for N_H and N_D. We can use the fact that the columns of T are unit vectors.

    # Generate a plausible range for the count rate
    x = np.arange(500, 5000, 1)
    s = np.empty(len(x))
    for i in range(len(x)):
        s[i] = np.linalg.norm(p_inverse * (2/x[i] * C_H - np.ones((n, 1))))
    fig, ax = plt.subplots()
    ax.plot(x, s)

    # The intersection of this and the curve should be at x = n_H
    plt.axhline(y=1)

    # The actual value of N_H
    plt.axvline(x=N_H)

    ax.set(xlabel='n_H', ylabel='Magnitude', title='Solving for n_H')
    ax.grid()

    plt.show()









