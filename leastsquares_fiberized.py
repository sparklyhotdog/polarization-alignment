import numpy as np
from scipy.spatial.transform import Rotation as r
from scipy.optimize import least_squares, direct, minimize
import random
from measurements import measure, generate_eulerangles
from nonideal import rotation_nonideal_axes, calculate_euler_angles
from plot_fringe import plot


def calc_expected_counts(rotations):
    # Generate arbitrary rotation matrices, T and F,
    # (3x3, not Mueller matrices so that we can use the scipy rotation library)
    # the thetas are the Euler angles of T (degrees)
    theta1, theta2, theta3 = random.randrange(0, 360), random.randrange(0, 180), random.randrange(0, 360)
    F = r.random().as_matrix()
    T = rotation_nonideal_axes(axes[0], axes[1], axes[2], [theta1, theta2, theta3], degrees=True)

    # theta and phi are the spherical angles for the first row of F. (radians)
    # We also need to adjust the range of theta, since the range for arctan is [-pi/2, pi.2], but we prefer [0, 2pi]
    theta = np.arctan(F[0, 1] / F[0, 0])  # [-pi/2, pi/2]
    if F[0, 0] < 0:
        theta += np.pi
    elif F[0, 1] < 0:
        theta += 2 * np.pi
    phi = np.arccos(F[0, 2])  # [0, pi]

    N_H = random.uniform(.2, 1)
    N_D = random.uniform(.2, 1)

    actual_x = np.asarray([theta1, theta2, theta3, theta, phi, N_H, N_D])

    sigma = .00       # standard deviation for noise
    generated_counts = np.empty(2 * num_rotations)
    for i in range(num_rotations):
        # Map C_H to the even indices and C_D to the odd ones
        generated_counts[2 * i] = random.gauss(0.5 * N_H * (1 + (
                    np.asmatrix([1, 0, 0]) * F * r.as_matrix(rotations)[i] * T *
                    np.asmatrix(np.asarray([[1], [0], [0]])))[0, 0]), sigma)

        generated_counts[2 * i + 1] = random.gauss(0.5 * N_D * (1 + (
                    np.asmatrix([1, 0, 0]) * F * r.as_matrix(rotations)[i] * T *
                    np.asmatrix(np.asarray([[0], [1], [0]])))[0, 0]), sigma)

    return generated_counts, actual_x


# var is a list [theta1, theta2, theta3, theta, phi, N_H, N_D]
# theta1, theta2, theta3 [0, 360] are the euler angles (xyx) in degrees
# theta [0, 2pi] and phi [0, pi] are the spherical angles for the first row of F
def residuals(var, count_data):
    res = np.empty(2 * num_rotations)
    for index in range(num_rotations):
        calc_T = rotation_nonideal_axes(axes[0], axes[1], axes[2], [var[0], var[1], var[2]], degrees=True)
        F_row1 = np.asmatrix(
            np.asarray([np.cos(var[3]) * np.sin(var[4]), np.sin(var[3]) * np.sin(var[4]), np.cos(var[4])]))
        calculated_C_H = 0.5 * var[5] * \
            (1 + F_row1 * r.as_matrix(rotation_list)[index] * calc_T * np.asmatrix(np.asarray([[1], [0], [0]])))[0, 0]
        calculated_C_D = 0.5 * var[6] * \
            (1 + F_row1 * r.as_matrix(rotation_list)[index] * calc_T * np.asmatrix(np.asarray([[0], [1], [0]])))[0, 0]

        res[2 * index] = calculated_C_H - count_data[2 * index]
        res[2 * index + 1] = calculated_C_D - count_data[2 * index + 1]
    return res


def cost(var, count_data):
    return np.sum(residuals(var, count_data)**2)


def least_squares_fitting(count_data):
    counts_reorganized = np.reshape(count_data, (2, num_rotations), order='F')
    max_C_H = max(counts_reorganized[0])
    max_C_D = max(counts_reorganized[1])
    bounds = ([0, 0, 0, 0, 0, max_C_H, max_C_D], [360, 180, 360, 2 * np.pi, np.pi, np.inf, np.inf])
    bounds_direct = [(0, 360), (0, 180), (0, 360), (0, 2 * np.pi), (0, np.pi), (max_C_H, 1.5 * max_C_H),
                     (max_C_D, 1.5 * max_C_D)]
    # x0 = [random.randrange(0, 360), random.randrange(0, 180), random.randrange(0, 360),
    #       random.uniform(0, 2 * np.pi), random.uniform(0, np.pi),
    #       max_C_H, max_C_D]
    initial_result = direct(cost, bounds_direct, args=(count_data,))
    x0 = initial_result.x
    fitting = least_squares(residuals, x0, bounds=bounds, max_nfev=500, ftol=1e-10, xtol=1e-9, verbose=1, args=(count_data,))
    return fitting


def calculate_ret_angles(var):
    ret_angles = np.zeros(6)

    ret_angles[0:3] = -np.flip(var[0:3]) % 360

    f_theta = var[3]
    f_phi = var[4]
    f_row1 = np.asarray([np.cos(f_theta) * np.sin(f_phi), np.sin(f_theta) * np.sin(f_phi), np.cos(f_phi)])
    ret_angles[3] = 360 - np.arccos(f_row1[0]) * 180 / np.pi
    ret_angles[4] = np.arccos(f_row1[2] / np.sqrt(f_row1[1] ** 2 + f_row1[2] ** 2)) * 180 / np.pi
    if f_row1[1] > 0:
        ret_angles[4] = 360 - ret_angles[4]
    print(ret_angles)

    return ret_angles


if __name__ == "__main__":
    num_rotations = 16
    # rotation_list = r.random(num_rotations)
    rotation_list = r.from_rotvec(np.asarray([[0, 0, 0],
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
    axes = np.asarray([[[.999633], [.0038151], [-.0268291]],
                       [[.0271085], [.999295], [.0259618]],
                       [[.9994289], [-.0335444], [.004005751]],
                       [[0], [1], [0]],
                       [[.997268], [-0.0702493], [0.0228234]],
                       [[-0.00005461419], [.999687], [-0.0250044]]])

    # counts, actual_x = calc_expected_counts(rotation_list)
    # print(actual_x)
    # print(rotation_nonideal_axes(axes[0], axes[1], axes[2], [actual_x[0], actual_x[1], actual_x[2]], degrees=True))
    # print(np.asarray([np.cos(actual_x[3]) * np.sin(actual_x[4]), np.sin(actual_x[3]) * np.sin(actual_x[4]), np.cos(actual_x[4])]))
    # counts = measure(num_rotations, generate_eulerangles(rotations=rotation_list), yaml_fn='serverinfo.yaml',
    #                  verbose=True, datapath='data/data.txt')
    counts = np.loadtxt('data/data.txt')
    # print(counts)

    result = least_squares_fitting(counts)

    x = result.x
    np.savetxt('data/leastsquares_output.txt', x)
    calculated_T = rotation_nonideal_axes(axes[0], axes[1], axes[2], [x[0], x[1], x[2]], degrees=True)
    print("\nCalculated T: \n", calculated_T)
    f = np.asmatrix(np.asarray([np.cos(x[3]) * np.sin(x[4]), np.sin(x[3]) * np.sin(x[4]), np.cos(x[4])]))
    print("First row of F: ", f)
    print("Result: ", x)
    print("Cost: ", result.cost)

    P = calculate_ret_angles(x)
    # plot(P, title=str(P), filepath='plots/figii.png')
