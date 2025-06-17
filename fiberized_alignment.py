import numpy as np
from scipy.spatial.transform import Rotation as r
import random
from scipy.optimize import direct, dual_annealing
import matplotlib.pyplot as plt
from matplotlib import cm


class FiberizedAlignment:
    def __init__(self, counts_H, counts_D, rotation_list, F_row1):
        self.rotation_list = rotation_list
        num_rotations = len(rotation_list)

        P = np.asmatrix(np.empty((num_rotations, 3)))
        # Construct our P matrix for all rotations. The ith row of P contains the first row of rotation matrix i.
        # We only care about the first row of each rotation, since we are multiplying it by the H polarizer matrix.

        for i in range(num_rotations):
            P[i] = F_row1 @ rotations[i]

        p_inverse = np.linalg.pinv(P)

        # We first solve for N_H and N_D.
        # We can set up a quadratic equation for both from the fact that the columns of T are unit vectors
        Pc_H = p_inverse * counts_H
        Pc_D = p_inverse * counts_D
        A1 = p_inverse * np.ones((num_rotations, 1))

        a = np.sum(np.square(A1)) - 1  # The a term is the same for both
        b_H = -4 * np.sum(np.multiply(Pc_H, A1))
        b_D = -4 * np.sum(np.multiply(Pc_D, A1))
        c_H = 4 * np.sum(np.square(Pc_H))
        c_D = 4 * np.sum(np.square(Pc_D))

        self.roots_H = np.roots([a, b_H, c_H])
        self.roots_D = np.roots([a, b_D, c_D])
        # print(self.roots_H, self.roots_D)

        # We have 2 possible answers for both N_H and N_D, which gives us 4 scenarios in total.
        self.possibleTs = np.empty((4, 3, 3))
        self.determinants = np.empty(4)
        self.dotproducts = np.empty(4)
        self.error = np.empty(4)

        for i in range(4):
            self.N_h = self.roots_H[i % 2].real
            self.N_d = self.roots_D[i // 2].real

            # Check if col1 and col2 must be orthogonal
            T_col1 = p_inverse * (2 / self.N_h * counts_H - np.ones((num_rotations, 1)))
            T_col2 = p_inverse * (2 / self.N_d * counts_D - np.ones((num_rotations, 1)))
            T_col3 = np.cross(T_col1.reshape(3), T_col2.reshape(3)).reshape((3, 1))

            # Put the columns of T together to get our rotation matrix
            self.possibleTs[i] = np.hstack((T_col1, T_col2, T_col3))

            self.dotproducts[i] = (T_col1.reshape((1, 3)) @ T_col2)[0, 0]         # should be 0
            # print("dot product between cols: ", self.dotproducts[i])

            self.determinants[i] = np.linalg.det(self.possibleTs[i])        # should be 1
            # print("det: ", self.determinants[i])

            calc_c_h, calc_c_d = calculate_counts(rotations, N_H, N_D, self.possibleTs[i], F_row1)

            self.error[i] = np.linalg.norm(counts_H - calc_c_h) + np.linalg.norm(counts_D - calc_c_d)
            # print("error: ", self.error[i])


def calculate_counts(rotation_list, N_H, N_D, T_matrix, F):
    """Helper function for testing/simulation to calculate the expected photon counts for both the H and D states,
    given a list of rotations, the count rates, N_H and N_D, and a transformation T"""
    C_H = np.asmatrix(np.empty((len(rotation_list), 1)))
    C_D = np.asmatrix(np.empty((len(rotation_list), 1)))
    for i in range(len(rotation_list)):
        if F.shape == (3, 3):
            C_H[i][0] = 0.5 * N_H * (
                    1 + np.asmatrix([1, 0, 0]) @ F @ rotation_list[i] @ T_matrix @ np.asmatrix(np.asarray([[1], [0], [0]])))
            C_D[i][0] = 0.5 * N_D * (
                    1 + np.asmatrix([1, 0, 0]) @ F @ rotation_list[i] @ T_matrix @ np.asmatrix(np.asarray([[0], [1], [0]])))
        else:
            C_H[i][0] = 0.5 * N_H * (1 + F @ rotation_list[i] @ T_matrix @ np.asmatrix(np.asarray([[1], [0], [0]])))
            C_D[i][0] = 0.5 * N_D * (1 + F @ rotation_list[i] @ T_matrix @ np.asmatrix(np.asarray([[0], [1], [0]])))

    return C_H, C_D


def graph():
    n = 10

    theta = np.linspace(0, 2 * np.pi, n)
    phi = np.linspace(0, np.pi, n)

    errors = np.empty((4, n, n))
    dets = np.empty((4, n, n))
    dots = np.empty((4, n, n))
    for i in range(n):
        for j in range(n):
            F = np.asmatrix(
                np.asarray([np.cos(theta[i]) * np.sin(phi[j]), np.sin(theta[i]) * np.sin(phi[j]), np.cos(phi[j])]))
            A = FiberizedAlignment(C_H, C_D, rotations, F)
            for k in range(4):
                errors[k][i][j] = A.error[k]
                dets[k][i][j] = A.determinants[k]
                dots[k][i][j] = A.dotproducts[k]

    theta, phi = np.meshgrid(theta, phi)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    i = 1
    color = cm.coolwarm(errors[i] / np.max(errors[i]))
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, facecolors=color, linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def fun(x, C_H, C_D, rotations):
    F = np.asmatrix(np.asarray([np.cos(x[0]) * np.sin(x[1]), np.sin(x[0]) * np.sin(x[1]), np.cos(x[1])]))
    A = FiberizedAlignment(C_H, C_D, rotations, F)
    return min(A.error)


if __name__ == "__main__":

    T = r.random().as_matrix()
    F = r.random().as_matrix()
    N_H = random.randrange(500, 1500)
    N_D = random.randrange(500, 1500)
    print("T: ", T)
    print("F: ", F[0])

    # Pick n >= 3 rotations for the PA to emulate
    R1 = r.as_matrix(r.from_rotvec([1, 0, 0]))
    R2 = r.as_matrix(r.from_rotvec([0, 1, 0]))
    R3 = r.as_matrix(r.from_rotvec([0, 0, 1]))
    R4 = r.as_matrix(r.from_rotvec([0, 1, 1]))
    rotations = [R1, R2, R3, R4]

    C_H, C_D = calculate_counts(rotations, N_H, N_D, T, F)

    # ----------------------------------

    bounds = [(0, 2*np.pi), (0, np.pi)]

    # res = direct(fun, bounds, args=(C_H, C_D, rotations), f_min=0)
    res = dual_annealing(fun, bounds, args=(C_H, C_D, rotations))
    F_guess = np.array([np.cos(res.x[0]) * np.sin(res.x[1]), np.sin(res.x[0]) * np.sin(res.x[1]), np.cos(res.x[1])])
    print("Calculated F:\n", F_guess)
    # print(res.fun)
    # print(res.message)

    A = FiberizedAlignment(C_H, C_D, rotations, F_guess)
    index = np.argmin(A.error)
    # print(A.determinants[index])
    # print(A.dotproducts[index])
    print("Calculated T: \n", A.possibleTs[index])


