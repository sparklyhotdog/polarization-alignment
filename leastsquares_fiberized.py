import numpy as np
from scipy.spatial.transform import Rotation as r
from scipy.optimize import least_squares, direct, minimize
import random
from measurements import measure_HD
from nonideal import rotation_nonideal_axes, calculate_euler_angles
from plot_fringe import plot, plot2


def residuals(var, count_data, rotation_list, axes=None):
    """"Given an array of our unknowns, the count measurements, an array of scipy rotations, and an optional array of
    non-ideal axes of rotation (nx3 array),
    returns the residuals for each measurement.

    var is an array representing [theta1, theta2, theta3, theta, phi, N_H, N_D], where
    theta1, theta2, theta3 [0, 360] are the euler angles (xyx, or around the nonideal axes if provided) in degrees, and
    theta [0, 2pi] and phi [0, pi] are the spherical angles for the first row of F"""
    num_rotations = len(rotation_list)
    res = np.empty(2 * num_rotations)
    for index in range(num_rotations):
        if axes is None:
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
                     (max_C_D, 1.5 * max_C_D)]
    initial_result = direct(cost, bounds_direct, args=(count_data, rotation_list, axes))
    x0 = initial_result.x
    fitting = least_squares(residuals, x0, bounds=bounds, max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-10, verbose=verbose, args=(count_data, rotation_list, axes))
    return fitting


def calc_ret_angles_from_x(var):
    """Returns the retardance angles (degrees) for the 6 wave plates to undo T and F, given the solution of the least-squares
    optimization."""
    ret_angles = np.zeros(6)

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
    ret_angles[0:3] = -np.flip(calculate_euler_angles(T, axes[0:3], degrees=True, error_threshold=0.00001)) % 360

    ret_angles[3] = 360 - np.arccos(F[0]) * 180 / np.pi
    ret_angles[4] = np.arccos(F[2] / np.sqrt(F[1] ** 2 + F[2] ** 2)) * 180 / np.pi
    if F[1] > 0:
        ret_angles[4] = 360 - ret_angles[4]

    return ret_angles



if __name__ == "__main__":
    # rotations = r.from_rotvec(np.asarray([[0, 0, 0],
    #                                       [1, 0, 0],
    #                                       [2, 0, 0],
    #                                       [3, 0, 0],
    #                                       [0, 1, 0],
    #                                       [0, 2, 0],
    #                                       [0, 3, 0],
    #                                       [0, 0, 1],
    #                                       [0, 0, 2],
    #                                       [0, 0, 3],
    #                                       [1, 1, 0],
    #                                       [2, 2, 0],
    #                                       [1, 0, 1],
    #                                       [2, 0, 2],
    #                                       [0, 1, 1],
    #                                       [1, 1, 1]]))
    rotations = r.random(20)
    nonideal_axes = np.asarray([[[.999633], [.0038151], [-.0268291]],
                                [[.0271085], [.999295], [.0259618]],
                                [[.9994289], [-.0335444], [.004005751]],
                                [[0], [1], [0]],
                                [[.997268], [-0.0702493], [0.0228234]],
                                [[-0.00005461419], [.999687], [-0.0250044]]])
    # counts, actual_x = generate_expected_counts(rotations)
    counts, angles = measure_HD(r.as_euler(rotations, "xyx", degrees=True), verbose=True, datapath='data/data.txt',
                                rotpath='data/rot_angles.txt')
    for i in range(len(rotations)):
        rotations[i] = r.from_matrix(rotation_nonideal_axes(nonideal_axes, angles[i], degrees=True))
    # counts = np.loadtxt('data/data.txt')

    result = least_squares_fitting(counts, rotations, axes=nonideal_axes)

    x = result.x
    np.savetxt('data/leastsquares_output.txt', x)

    calculated_T = rotation_nonideal_axes(nonideal_axes[0:3], x[0:3], degrees=True)
    calculated_F = np.asarray([np.cos(x[3]) * np.sin(x[4]), np.sin(x[3]) * np.sin(x[4]), np.cos(x[4])])
    print("\nCalculated T: \n", calculated_T)
    print("First row of F: ", calculated_F)
    print("Result: ", x)
    print("Cost: ", result.cost)

    avg = (x[5] + x[6]) / 2
    count_rates = [x[5], avg, x[6], avg]

    P_1 = calc_ret_angles_from_x(x)
    print("Angles: ", P_1)
    calculated_T[:, 0:2] = -calculated_T[:, 0:2]
    calculated_F = -calculated_F
    print("Other T: ", calculated_T)
    print("Other F: ", calculated_F)
    P_2 = calc_ret_angles_from_matrix(calculated_T, calculated_F, nonideal_axes)
    plot2([P_1, P_2], title=str(np.round(P_1, 2)) + '\n' + str(np.round(P_2, 2)), filepath='plots/jun30_.png', expected_count_rates=None)

