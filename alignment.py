import numpy as np
from scipy.spatial.transform import Rotation as r
import random
'''
Polarization alignment process for a free-space polarization analyzer. 

Reference signals (H or D) are sent. They first go through an unknown fiber transformation, represented by rotation 
matrix T, then through the polarization analyzer whose rotations we can control, and lastly a H polarizer and detector. 
From these measured count rates, and our choice of PA rotations, our goal is to determine T.

'''


def calculate_counts(rotation_list, N_H, N_D, T):
    """Calculate the expected photon counts for both the H and D states, given a list of rotations, the count rates,
    N_H and N_D, and a transformation T"""
    C_H = np.asmatrix(np.empty((n, 1)))
    C_D = np.asmatrix(np.empty((n, 1)))
    for i in range(n):
        C_H[i][0] = 0.5 * N_H * (
                1 + np.asmatrix([1, 0, 0]) * rotation_list[i] * T * np.asmatrix(np.asarray([[1], [0], [0]])))
        C_D[i][0] = 0.5 * N_D * (
                1 + np.asmatrix([1, 0, 0]) * rotation_list[i] * T * np.asmatrix(np.asarray([[0], [1], [0]])))
    return C_H, C_D


if __name__ == "__main__":
    count = 0
    for j in range(100):
        # Generate an arbitrary rotation matrix (3x3, not a Mueller matrix so that we can use the scipy rotation library)
        T = r.as_matrix(r.from_euler("xyz", [2*np.pi*random.random(), 2*np.pi*random.random(), 2*np.pi*random.random()]))
        N_H = random.randrange(500, 1500)
        N_D = random.randrange(500, 1500)

        # Pick n >= 3 rotations for the PA to emulate
        # TODO: find a specific condition for this. not all sequences of rotations will work (something pseudoinverse exists blah blah)
        R1 = r.as_matrix(r.from_rotvec([1, 0, 0]))
        R2 = r.as_matrix(r.from_rotvec([0, 1, 0]))
        R3 = r.as_matrix(r.from_rotvec([0, 0, 1]))
        rotations = [R1, R2, R3]
        n = len(rotations)

        C_H, C_D = calculate_counts(rotations, N_H, N_D, T)
        P = np.asmatrix(np.empty((n, 3)))
        # Construct our P matrix for all rotations. The ith row of P contains the first row of rotation matrix i.
        # We only care about the first row of each rotation, since we are multiplying it by the H polarizer matrix.
        for i in range(n):
            P[i] = rotations[i][0]

        # Setup done ---------------------------------------------------------------------------------------------------
        # We have an unknown rotation T, and measurements C_H and C_D corresponding to the chosen rotations
        p_inverse = np.linalg.pinv(P)

        # We first solve for N_H and N_D.
        # We can set up a quadratic equation for both from the fact that the columns of T are unit vectors
        Pc_H = p_inverse * C_H
        Pc_D = p_inverse * C_D
        A1 = p_inverse * np.ones((n, 1))

        a = np.sum(np.square(A1)) - 1       # The a term is the same for both
        b_H = -4 * np.sum(np.multiply(Pc_H, A1))
        b_D = -4 * np.sum(np.multiply(Pc_D, A1))
        c_H = 4 * np.sum(np.square(Pc_H))
        c_D = 4 * np.sum(np.square(Pc_D))

        roots_H = np.roots([a, b_H, c_H])
        roots_D = np.roots([a, b_D, c_D])

        # We have 2 possible answers for both N_H and N_D, which gives us 4 scenarios in total.
        # However, the extraneous solutions won't produce valid rotation matrices (orthogonal matrices with det 1)
        # So we iterate through the 4 scenarios and stop when we find a valid rotation matrix.
        found_T = False
        calculated_T = np.empty((3, 3))
        for i in range(4):
            n_h = roots_H[i % 2]
            n_d = roots_D[i // 2]

            # Check if col1 and col2 must be orthogonal
            T_col1 = p_inverse * (2 / n_h * C_H - np.ones((n, 1)))
            T_col2 = p_inverse * (2 / n_d * C_D - np.ones((n, 1)))

            if np.dot(T_col1.reshape(3), T_col2) > 1e-8:
                continue

            # Compute the 3rd column by taking the cross-product of the first 2 columns
            T_col3 = np.cross(T_col1.reshape(3), T_col2.reshape(3)).reshape((3, 1))

            # Put the columns of T together to get our rotation matrix
            calculated_T = np.hstack((T_col1, T_col2, T_col3))

            # Check if det(T) equals 1
            if abs(np.linalg.det(calculated_T) - 1) > 1e-8:
                continue

            found_T = True
            break

        if not found_T:
            print("Error :(")

        error = np.linalg.norm(T - calculated_T)

        if error < 0.001:
            count += 1

        # print("N_H, N_D: ", (N_H, N_D))
        # print("Calculated n_H, n_D: ", n_h, n_d)
        # print("\nT:\n", T)
        # print("\nCalculated T:\n", calculated_T)

    print(count)


