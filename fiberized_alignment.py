import numpy as np
from scipy.spatial.transform import Rotation as r
import random
from scipy.optimize import direct, dual_annealing, minimize
from measurements import measure, generate_eulerangles
from nonideal import rotation_nonideal_axes, calculate_euler_angles
import matplotlib.pyplot as plt
from matplotlib import cm


class FiberizedAlignment:
    def __init__(self, counts_H, counts_D, rotation_matrices):

        calculated_F = find_F(counts_H, counts_D, rotation_matrices)
        Ts, errors = calculate_T_from_F(counts_H, counts_D, rotation_matrices, calculated_F)
        index = np.argmin(errors)
        calculated_T = Ts[index]

        # check that T is a rotation matrix
        det = np.linalg.det(calculated_T)
        dot = np.dot(calculated_T[:, 0], calculated_T[:, 1])
        is_rot_mat = (abs(det - 1) < 1e-5 and dot < 1e-5)
        # print("determinant, dot product: ", det, dot)

        if not is_rot_mat:
            print("Not a rotation matrix :(")

        self.F = calculated_F
        self.T = calculated_T


def find_F(counts_H, counts_D, rotation_matrices):
    """Finds F by minimizing the error function. First finds an estimate using a global minimum algorithm, then uses
    that result as the initial guess for a local minima finder."""

    bounds = [(0, 2 * np.pi), (0, np.pi)]

    initial_result = direct(error, bounds, args=(counts_H, counts_D, rotation_matrices), f_min=0)
    # initial_result = dual_annealing(error, bounds, args=(counts_H, counts_D, rotation_matrices))
    F_guess = initial_result.x
    result = minimize(error, F_guess, args=(counts_H, counts_D, rotation_matrices), bounds=bounds)
    x = result.x
    # print(F_guess, x)

    return np.array([np.cos(x[0]) * np.sin(x[1]), np.sin(x[0]) * np.sin(x[1]), np.cos(x[1])])


def calculate_T_from_F(counts_H, counts_D, rotation_matrices, F_row1):
    """Given the first row of F, calculates the 4 possible T matrices. If F is not correct, it may not output rotation
    matrices."""
    num_rotations = rotation_matrices.shape[0]

    P = np.asmatrix(np.empty((num_rotations, 3)))
    # Construct our P matrix for all rotations. The ith row of P contains the first row of rotation matrix i.

    for i in range(num_rotations):
        P[i] = F_row1 @ rotation_matrices[i]

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

    roots_H = np.roots([a, b_H, c_H])
    roots_D = np.roots([a, b_D, c_D])

    # We have 2 possible answers for both N_H and N_D, which gives us 4 scenarios in total.
    possibleTs = np.empty((4, 3, 3))
    errors = np.empty(4)

    for i in range(4):
        N_h = roots_H[i % 2].real
        N_d = roots_D[i // 2].real

        # Check if col1 and col2 must be orthogonal
        T_col1 = p_inverse * (2 / N_h * counts_H - np.ones((num_rotations, 1)))
        T_col2 = p_inverse * (2 / N_d * counts_D - np.ones((num_rotations, 1)))
        T_col3 = np.cross(T_col1.reshape(3), T_col2.reshape(3)).reshape((3, 1))

        # Put the columns of T together to get our rotation matrix
        possibleTs[i] = np.hstack((T_col1, T_col2, T_col3))

        calc_c_h, calc_c_d = calculate_counts(rotation_matrices, N_h, N_d, possibleTs[i], F_row1)

        errors[i] = np.linalg.norm(counts_H - calc_c_h) + np.linalg.norm(counts_D - calc_c_d)

    return possibleTs, errors


def error(x, counts_H, counts_D, rotation_matrices):
    """The error function we are trying to minimize for finding F.
    x is an array of theta and phi in radians--the spherical angles for the first row of F"""
    F = np.asmatrix(np.asarray([np.cos(x[0]) * np.sin(x[1]), np.sin(x[0]) * np.sin(x[1]), np.cos(x[1])]))
    Ts, errors = calculate_T_from_F(counts_H, counts_D, rotation_matrices, F)
    return min(errors)


def calculate_counts(rotation_matrices, N_H, N_D, T_matrix, F):
    """Helper function to calculate the expected photon counts for both the H and D states,
    given a list of rotations, the count rates, N_H and N_D, a rotation matrix T, and the first row of the F matrix"""
    C_H = np.asmatrix(np.empty((len(rotation_matrices), 1)))
    C_D = np.asmatrix(np.empty((len(rotation_matrices), 1)))
    for i in range(len(rotation_matrices)):
        if F.shape == (3, 3):
            C_H[i][0] = 0.5 * N_H * (
                    1 + np.asmatrix([1, 0, 0]) @ F @ rotation_matrices[i] @ T_matrix @ np.asmatrix(np.asarray([[1], [0], [0]])))
            C_D[i][0] = 0.5 * N_D * (
                    1 + np.asmatrix([1, 0, 0]) @ F @ rotation_matrices[i] @ T_matrix @ np.asmatrix(np.asarray([[0], [1], [0]])))
        else:
            C_H[i][0] = 0.5 * N_H * (1 + F @ rotation_matrices[i] @ T_matrix @ np.asmatrix(np.asarray([[1], [0], [0]])))
            C_D[i][0] = 0.5 * N_D * (1 + F @ rotation_matrices[i] @ T_matrix @ np.asmatrix(np.asarray([[0], [1], [0]])))

    return C_H, C_D


def generate_counts(rotation_matrices):
    T = r.random().as_matrix()
    F = r.random().as_matrix()
    N_H = random.randrange(500, 1500)
    N_D = random.randrange(500, 1500)
    print("T: ", T)
    print("F: ", F[0])

    return calculate_counts(rotation_matrices, N_H, N_D, T, F)


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
            A = FiberizedAlignment(C_H, C_D, rotation_matrices, F)
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



if __name__ == "__main__":

    # Pick n >= 3 rotations for the PA to emulate
    rotations = r.from_rotvec(np.array([[0, 0, 0],
                                        [1, 0, 0],
                                        [2, 0, 0],
                                        [3, 0, 0],
                                        [0, 1, 0],
                                        [0, 2, 0],
                                        [0, 3, 0],
                                        [0, 0, 1],
                                        [0, 0, 2],
                                        [0, 0, 3],
                                        [1, 1, 0],
                                        [2, 2, 0],
                                        [1, 0, 1],
                                        [2, 0, 2],
                                        [0, 1, 1],
                                        [1, 1, 1]]))
    rots = rotations.as_matrix()

    C_H, C_D = generate_counts(rots)
    # counts = measure(len(rotations), generate_eulerangles(rotations=rotations), yaml_fn='serverinfo.yaml',
    #                  verbose=True, datapath='data/alignment_data.txt')
    # counts = np.loadtxt('data/alignment_data.txt')
    # counts_reorganized = np.reshape(counts, (2, len(rotations)), order='F')
    # C_H, C_D = np.reshape(counts_reorganized[0], (len(rotations), 1)), np.reshape(counts_reorganized[1], (len(rotations), 1))

    # ----------------------------------
    A = FiberizedAlignment(C_H, C_D, rots)
    print("Calculated T:\n", A.T)
    print("Calculated F:\n", A.F)



