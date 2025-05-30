import numpy as np
from scipy.spatial.transform import Rotation as r
import random
import matplotlib.pyplot as plt
'''
Polarization alignment process for a free-space polarization analyzer. 

Reference signals (H or D) are sent. They first go through an unknown fiber transformation, represented by rotation 
matrix T, then through the polarization analyzer whose rotations we can control, and lastly a H polarizer and detector. 
From these measured count rates, and our choice of PA rotations, our goal is to determine T.

'''

if __name__ == "__main__":
    # Generate an arbitrary rotation matrix (3x3, not a Mueller matrix so that we can use the scipy rotation library)
    T = r.as_matrix(r.from_euler("xyz", [2*np.pi*random.random(), 2*np.pi*random.random(), 2*np.pi*random.random()]))
    N_H = random.randrange(500, 1500)
    N_D = random.randrange(500, 1500)

    # Pick n >= 3 rotations for the PA to emulate
    # TODO: find a specific condition for this. not all sequences of rotations will work (something pseudomatrix exists blah blah)
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

    # Solve for N_H and N_D. We can use the fact that the columns of T are unit vectors, and solve a quadratic for both.
    Pc_H = p_inverse * C_H
    Pc_D = p_inverse * C_D
    A1 = p_inverse * np.ones((n, 1))

    a = np.sum(np.square(A1)) - 1
    b_H = -4 * np.sum(np.multiply(Pc_H, A1))
    b_D = -4 * np.sum(np.multiply(Pc_D, A1))
    c_H = 4 * np.sum(np.square(Pc_H))
    c_D = 4 * np.sum(np.square(Pc_D))

    # Seems like the right answer is usually the lower one, but probably need to check both
    # TODO: check the other one, also maybe round to nearest int?
    n_H = min(np.roots([a, b_H, c_H]))
    n_D = min(np.roots([a, b_D, c_D]))

    T_col1 = p_inverse * (2 / n_H * C_H - np.ones((n, 1)))
    T_col2 = p_inverse * (2 / n_D * C_D - np.ones((n, 1)))

    # We can get the 3rd column by taking the cross-product of the first 2 columns, since we know that T is a rotation
    # matrix and has orthonormal columns.
    T_col3 = np.cross(T_col1.reshape(3), T_col2.reshape(3)).reshape((3, 1))

    # Put the columns of T together
    calculated_T = np.hstack((T_col1, T_col2, T_col3))

    print("N_H, N_D: ", (N_H, N_D))
    print("Calculated n_H, n_D: ", (n_H, n_D))

    print("\nT:\n", T)
    print("\nCalculated T:\n", calculated_T)



