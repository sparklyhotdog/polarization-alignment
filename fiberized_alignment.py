import numpy as np
from scipy.spatial.transform import Rotation as r
import random
from scipy.optimize import direct, minimize
from measurements import measure_HD
from nonideal import rotation_nonideal_axes, calculate_euler_angles
from plot_fringe import plot, plot2


class FiberizedAlignment:
    def __init__(self, count_data, rotation_matrices):
        """Calculates the T matrix, the first row of F, and the count rates for the H and D states, respectively,
        given the count measurements and the associated array of rotation matrices"""
        counts_reorganized = np.reshape(count_data, (2, len(rotations)), order='F')
        counts_H, counts_D = (np.reshape(counts_reorganized[0], (len(rotations), 1)),
                              np.reshape(counts_reorganized[1], (len(rotations), 1)))
        calculated_F = find_F(counts_H, counts_D, rotation_matrices)
        Ts, errors, roots_H, roots_D = calculate_T_from_F(counts_H, counts_D, rotation_matrices, calculated_F)[0:4]
        index = np.argmin(errors)
        calculated_T = Ts[index]
        print("Error: ", errors[index])

        # check that T is a rotation matrix
        det = np.linalg.det(calculated_T)
        dot = np.dot(calculated_T[:, 0], calculated_T[:, 1])
        is_rot_mat = (abs(det - 1) < 1e-5 and dot < 1e-5)

        if not is_rot_mat:
            print("Determinant, dot product: ", det, dot)
            print("Not a rotation matrix :(")

        self.F = calculated_F
        self.T = calculated_T
        self.N_H = roots_H[index % 2]
        self.N_D = roots_D[index // 2]

    def calculate_retardance_angles(self, axes=None):
        """Returns the retardance angles to set the 6 wave plates in order to reverse T and F."""
        T_rot = r.from_matrix(self.T)
        ret_angles = np.zeros(6)

        # Assume ideal case
        if axes is None:
            print(T_rot.as_matrix())
            ret_angles[0:3] = -np.flip(T_rot.as_euler("xyx", degrees=True)) % 360

            ret_angles[3] = 360 - np.arccos(self.F[0]) * 180 / np.pi
            ret_angles[4] = np.arccos(self.F[2] / np.sqrt(self.F[1] ** 2 + self.F[2] ** 2)) * 180 / np.pi
            if self.F[1] > 0:
                ret_angles[4] = 360 - ret_angles[4]
            print(ret_angles)

        return ret_angles


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
    matrices. counts_H and counts_D are column vectors of the measurements.

    Returns length 4 arrays with the possible T matrices, errors, roots for N_H, roots for N_D, the determinants of T,
    and the dot products between col1 and col2 in T."""
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
    determinants = np.empty(4)
    dotproducts = np.empty(4)

    for i in range(4):
        N_h = roots_H[i % 2].real
        N_d = roots_D[i // 2].real

        # Check if col1 and col2 must be orthogonal
        T_col1 = p_inverse * (2 / N_h * counts_H - np.ones((num_rotations, 1)))
        T_col2 = p_inverse * (2 / N_d * counts_D - np.ones((num_rotations, 1)))
        T_col3 = np.cross(T_col1.reshape(3), T_col2.reshape(3)).reshape((3, 1))

        # Put the columns of T together to get our rotation matrix
        possibleTs[i] = np.hstack((T_col1, T_col2, T_col3))
        dotproducts[i] = (T_col1.reshape((1, 3)) @ T_col2)[0, 0]
        determinants[i] = np.linalg.det(possibleTs[i])

        calc_c_h, calc_c_d = calculate_counts(rotation_matrices, N_h, N_d, possibleTs[i], F_row1)

        errors[i] = np.linalg.norm(counts_H - calc_c_h) + np.linalg.norm(counts_D - calc_c_d)

    return possibleTs, errors, roots_H, roots_D, determinants, dotproducts


def error(x, counts_H, counts_D, rotation_matrices):
    """The error function we are trying to minimize for finding F.

    Returns the error, given x (the spherical angles for the first row of F in radians),
    counts_H and counts_D (column vectors of the measurements),
    and rotation_matrices (an array of rotation matrices)"""
    F = np.asmatrix(np.asarray([np.cos(x[0]) * np.sin(x[1]), np.sin(x[0]) * np.sin(x[1]), np.cos(x[1])]))
    errors = calculate_T_from_F(counts_H, counts_D, rotation_matrices, F)[1]
    return min(errors)


def calculate_counts(rotation_matrices, N_H, N_D, T_matrix, F):
    """Helper function to calculate the expected photon counts for both the H and D states,
    given an array of rotation matrices, the count rates (N_H and N_D) a rotation matrix T, and the first row of the F matrix"""
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


def generate_counts(rotation_matrices, sigma=0.0):
    """Randomly generates T, F, N_H, N_D, and simulates the rotations and measurements. Returns the expected count data."""
    T = r.random().as_matrix()
    F = r.random().as_matrix()
    N_H = random.uniform(0.5, 1)
    N_D = random.uniform(0.5, 1)
    print("T: ", T)
    print("F: ", F[0])
    print("N_H, N_D:", N_H, N_D)

    counts_H, counts_D = calculate_counts(rotation_matrices, N_H, N_D, T, F)

    for i in range(counts_H.shape[0]):
        counts_H += random.gauss(sigma=sigma)
        counts_D += random.gauss(sigma=sigma)

    # print(calculate_T_from_F(counts_H, counts_D, rotation_matrices, F[0]))
    counts = np.hstack((counts_H, counts_D))
    counts = np.reshape(counts, 2*counts_H.shape[0])
    # TODO: convert counts_H and counts_D into a 1D array
    return counts


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
    nonideal_axes = np.asarray([[[.999633], [.0038151], [-.0268291]],
                                [[.0271085], [.999295], [.0259618]],
                                [[.9994289], [-.0335444], [.004005751]],
                                [[0], [1], [0]],
                                [[.997268], [-0.0702493], [0.0228234]],
                                [[-0.00005461419], [.999687], [-0.0250044]]])

    # counts = generate_counts(rots, sigma=0.001)
    counts, angles = measure_HD(r.as_euler(rotations, "xyx", degrees=True), verbose=True, datapath='data/data.txt')
    # counts = np.loadtxt('data/data.txt')

    for i in range(len(rotations)):
        rotations[i] = r.from_matrix(rotation_nonideal_axes(nonideal_axes, angles[i], degrees=True))
    rots = rotations.as_matrix()

    A = FiberizedAlignment(counts, rots)
    print("Calculated T:\n", A.T)
    print("Calculated F:\n", A.F)
    print("N_H, N_D: ", A.N_H, A.N_D)
    P = A.calculate_retardance_angles()
    plot(P, title=str(P), filepath='plots/jun30--.png')



