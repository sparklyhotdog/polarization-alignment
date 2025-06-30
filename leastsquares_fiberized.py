import numpy as np
from scipy.spatial.transform import Rotation as r
from scipy.optimize import least_squares, direct, minimize
import random
from measurements import measure_HD
from nonideal import rotation_nonideal_axes, calculate_euler_angles
from plot_fringe import plot, plot2


class Fiberized:
    # 6 axes of rotation of each of the H-D-H-D-H-D waveplates in the Nucrypt PA (taken from the flash of the PA)
    axes = np.asarray([[[.999633], [.0038151], [-.0268291]],
                      [[.0271085], [.999295], [.0259618]],
                      [[.9994289], [-.0335444], [.004005751]],
                      [[0], [1], [0]],
                      [[.997268], [-0.0702493], [0.0228234]],
                      [[-0.00005461419], [.999687], [-0.0250044]]])

    def __init__(self, rotation_list=None, verbose=True):
        """ Performs the "one-shot" alignment and calculates the T and F matrices, and the retardance angles to set to
        reverse them.

        rotation_list, if specified, should be a scipy rotation object containing multiple rotations.
        (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html)"""
        if rotation_list is None:
            rotation_list = r.random(16)
        self.counts, angles = measure_HD(r.as_euler(rotation_list, "xyx", degrees=True), verbose=verbose)

        # Update the rotations based on the angles given back by the polarization analyzer
        self.rotations = r.identity(len(rotation_list))
        for i in range(len(rotation_list)):
            self.rotations[i] = r.from_matrix(rotation_nonideal_axes(Fiberized.axes, angles[i], degrees=True))

        # The least squares solves for x [theta1, theta2, theta3, theta, phi, N_H, N_D], where
        # theta1, theta2, theta3 [0, 360] are the angles around the rotation axes (degrees),
        # theta [0, 2pi] and phi [0, pi] are the spherical angles for the first row of the F matrix (radians),
        # and N_H and N_D are the count rates for the H and D states (mW)
        self.least_squares_result = least_squares_fitting(self.counts, self.rotations, axes=Fiberized.axes)
        x = self.least_squares_result.x

        self.T = rotation_nonideal_axes(Fiberized.axes[0:3], x[0:3], degrees=True)
        self.F = np.asarray([np.cos(x[3]) * np.sin(x[4]), np.sin(x[3]) * np.sin(x[4]), np.cos(x[4])])
        self.N_H = x[5]
        self.N_D = x[6]

        self.other_T = np.copy(self.T)
        self.other_T[:, 0:2] = -self.T[:, 0:2]
        self.other_F = -self.F

        self.ret_angles = [calc_ret_angles_from_x(x), calc_ret_angles_from_matrix(self.other_T, self.other_F, Fiberized.axes)]

    def print_results(self):
        """Prints the calculated T and F matrices, the count rates of the H and D states, the cost of the fitting,
        and the calculated retardance angles to reverse T and F"""
        print("Calculated T: \n", self.T)
        print("Calculated first row of F: \n", self.F)
        print("N_H, N_D: ", self.N_H, self.N_D)
        print("Cost: ", self.least_squares_result.cost)
        print("Retardance angles: \n", self.ret_angles[0], '\n', self.ret_angles[1])

    def plot_fringes(self, filepath=None, verbose=False):
        """Plots the fringes with compensation"""
        angles_str = str(np.round(self.ret_angles[0], 2)) + '\n' + str(np.round(self.ret_angles[1], 2))
        plot2(ret_angles=self.ret_angles, title=angles_str, filepath=filepath, verbose=verbose)


def residuals(var, count_data, rotation_list, axes=None):
    """"Returns the residuals for each measurement, given an array of our unknowns, the count measurements, a scipy
    rotation object containing the rotations, and an optional array of non-ideal axes of rotation (nx3 array).

    var is an array representing [theta1, theta2, theta3, theta, phi, N_H, N_D], where
    theta1, theta2, theta3 [0, 360] are the euler angles (xyx, or around the nonideal axes if provided) in degrees, and
    theta [0, 2pi] and phi [0, pi] are the spherical angles for the first row of F"""
    num_rotations = len(rotation_list)
    res = np.empty(2 * num_rotations)
    for index in range(num_rotations):
        if axes is None:
            # Assume ideal rotation axes and use the angles as euler angles
            calc_T = r.as_matrix(r.from_euler("xyx", [var[0], var[1], var[2]], degrees=True))
        else:
            calc_T = rotation_nonideal_axes(axes[0:3], var[0:3], degrees=True)

        F_row1 = np.asmatrix(
            np.asarray([np.cos(var[3]) * np.sin(var[4]), np.sin(var[3]) * np.sin(var[4]), np.cos(var[4])]))

        calculated_C_H = 0.5 * var[5] * \
            (1 + F_row1 * r.as_matrix(rotation_list)[index] * calc_T * np.asmatrix(np.asarray([[1], [0], [0]])))[0, 0]
        calculated_C_D = 0.5 * var[6] * \
            (1 + F_row1 * r.as_matrix(rotation_list)[index] * calc_T * np.asmatrix(np.asarray([[0], [1], [0]])))[0, 0]

        res[2 * index] = calculated_C_H - count_data[2 * index]
        res[2 * index + 1] = calculated_C_D - count_data[2 * index + 1]
    return res


def cost(var, count_data, rotation_list, axes=None):
    """Converts the residuals into a scalar (the cost)"""
    return np.sum(residuals(var, count_data, rotation_list, axes=axes)**2)


def least_squares_fitting(count_data, rotation_list, axes=None, verbose=False):
    """Performs the least squares fitting to figure out the T matrix, the first row of the F matrix, and the count rates
     for the H and D states, respectively. Returns the scipy OptimizeResult object."""
    num_rotations = len(rotation_list)
    counts_reorganized = np.reshape(count_data, (2, num_rotations), order='F')
    max_C_H = max(counts_reorganized[0])
    max_C_D = max(counts_reorganized[1])
    bounds = ([0, 0, 0, 0, 0, max_C_H, max_C_D], [360, 180, 360, 2 * np.pi, np.pi, np.inf, np.inf])
    bounds_direct = [(0, 360), (0, 180), (0, 360), (0, 2 * np.pi), (0, np.pi), (max_C_H, 1.5 * max_C_H),
                     (max_C_D, 1.5 * max_C_D)]      # the direct method used a different bounds formatting

    # first calculate an initial guess using a global optimization algorithm
    initial_result = direct(cost, bounds_direct, args=(count_data, rotation_list, axes))
    x0 = initial_result.x
    # then use the least-squares algorithm with our initial guess to find a more exact solution
    fitting = least_squares(residuals, x0, bounds=bounds, max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-10, verbose=verbose, args=(count_data, rotation_list, axes))
    return fitting


def calc_ret_angles_from_x(var):
    """Returns the retardance angles (degrees) for the 6 wave plates to undo T and F, given the solution of the least-squares
    optimization."""
    ret_angles = np.zeros(6)

    # TODO: Fix this?? not always true, no?
    ret_angles[0:3] = -np.flip(var[0:3]) % 360

    # TODO: figure out angles for F given non-ideal axes of rotation
    f_theta = var[3]
    f_phi = var[4]
    f_row1 = np.asarray([np.cos(f_theta) * np.sin(f_phi), np.sin(f_theta) * np.sin(f_phi), np.cos(f_phi)])
    ret_angles[3] = 360 - np.arccos(f_row1[0]) * 180 / np.pi
    ret_angles[4] = np.arccos(f_row1[2] / np.sqrt(f_row1[1] ** 2 + f_row1[2] ** 2)) * 180 / np.pi
    if f_row1[1] > 0:
        ret_angles[4] = 360 - ret_angles[4]

    return ret_angles


def calc_ret_angles_from_matrix(T, F, axes):
    """Returns the retardance angles (degrees) for the 6 wave plates to undo T and F, given the 3x3 rotation matrix T
    and the first row of the matrix F"""
    ret_angles = np.zeros(6)
    # We want the first 3 wave plates to emulate T inverse.
    # Since T is a rotation matrix, it is orthogonal, and its inverse is equal to its transpose.
    # TODO: Fix this. the axes are flipped
    # ret_angles[0:3] = calculate_euler_angles(np.linalg.inv(T), axes[0:3], degrees=True, error_threshold=0.00001)
    ret_angles[0:3] = -np.flip(calculate_euler_angles(T, axes[0:3], degrees=True, error_threshold=0.00001)) % 360

    # The 4th and 5th wave plates undo F
    ret_angles[3] = 360 - np.arccos(F[0]) * 180 / np.pi
    ret_angles[4] = np.arccos(F[2] / np.sqrt(F[1] ** 2 + F[2] ** 2)) * 180 / np.pi
    if F[1] > 0:
        ret_angles[4] = 360 - ret_angles[4]

    return ret_angles


if __name__ == "__main__":
    A = Fiberized(verbose=False)
    A.print_results()
    # plot(title='No Compensation', filepath='plots/jun30_nocompensation.png')
    A.plot_fringes(filepath='plots/jun30_1.png', verbose=True)

