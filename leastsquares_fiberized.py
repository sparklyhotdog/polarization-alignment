import numpy as np
from scipy.spatial.transform import Rotation as r
from scipy.optimize import least_squares
import random


# var is a list [theta1, theta2, theta3, theta, phi, N_H, N_D]
# theta1, theta2, theta3 [0, 360] are the euler angles (xyx) in degrees
# theta [0, 2pi] and phi [0, pi] are the spherical angles for the first row of F
def residuals(var):
    res = np.empty(2*num_rotations)
    for index in range(num_rotations):
        calculated_T = r.from_euler("xyx", [var[0], var[1], var[2]], degrees=True).as_matrix()
        F_row1 = np.asmatrix(np.asarray([np.cos(var[3])*np.sin(var[4]), np.sin(var[3])*np.sin(var[4]), np.cos(var[4])]))
        calculated_C_H = 0.5 * var[5] * (1 + F_row1 * rotation_list[index] * calculated_T * np.asmatrix(np.asarray([[1], [0], [0]])))[0, 0]
        calculated_C_D = 0.5 * var[6] * (1 + F_row1 * rotation_list[index] * calculated_T * np.asmatrix(np.asarray([[0], [1], [0]])))[0, 0]

        res[2*index] = calculated_C_H - counts[2*index]
        res[2*index + 1] = calculated_C_D - counts[2*index + 1]
    return res


for index in range(10):
    # Generate arbitrary rotation matrices, T and F,
    # (3x3, not Mueller matrices so that we can use the scipy rotation library)
    # theta1, theta2, theta3 = random.randrange(0, 360), random.randrange(0, 180), random.randrange(0, 360)
    # F = r.random().as_matrix()
    theta1, theta2, theta3 = 5, 152, 67
    T = r.from_euler("xyx", [theta1, theta2, theta3], degrees=True).as_matrix()
    F = np.asmatrix([[0.42698964, -0.51922936, -0.74032474], [-0.57171132, -0.78932544,  0.223856], [-0.70058976,  0.32766784, -0.63388309]])

    # theta and phi are the spherical angles for the first row of F.
    # We also need to adjust the range of theta, since the range for arctan is [-pi/2, pi.2], but we prefer [0, 2pi]
    theta = np.arctan(F[0, 1]/F[0, 0])      # [-pi/2, pi/2]
    if F[0, 0] < 0:
        theta += np.pi
    elif F[0, 1] < 0:
        theta += 2 * np.pi
    phi = np.arccos(F[0, 2])                # [0, pi]

    # print(F)
    # print(np.asmatrix(np.asarray([np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)])))
    # print("theta, phi: ", theta, phi)
    # N_H = random.randrange(500, 1500)
    # N_D = random.randrange(500, 1500)
    N_H = 885
    N_D = 1115

    actual_x = np.asarray([theta1, theta2, theta3, theta, phi, N_H, N_D])

    # Generate random rotations
    num_rotations = 16
    rotations = r.random(num_rotations)
    rotation_list = rotations.as_matrix()

    # Calculate the expected counts -------------------------------------------------------
    sigma = 5
    counts = np.empty(2*num_rotations)
    for i in range(num_rotations):
        # Map C_H to the even indices and C_D to the odd ones
        counts[2 * i] = random.gauss(0.5 * N_H * (
                1 + (np.asmatrix([1, 0, 0]) * F * rotation_list[i] * T * np.asmatrix(np.asarray([[1], [0], [0]])))[0, 0]), sigma)
        counts[2*i + 1] = random.gauss(0.5 * N_D * (
                1 + (np.asmatrix([1, 0, 0]) * F * rotation_list[i] * T * np.asmatrix(np.asarray([[0], [1], [0]])))[0, 0]), sigma)

    # -------------------------------------------------------------------------------------
    cost_threshold = 1000
    bounds = ([0, 0, 0, 0, 0, 0, 0], [360, 180, 360, 2*np.pi, np.pi, np.inf, np.inf])
    print("\n")
    success = False
    while not success:
        x0 = [random.randrange(0, 360), random.randrange(0, 180), random.randrange(0, 360),
              random.uniform(0, 2*np.pi), random.uniform(0, np.pi),
              random.randrange(0, 2000), random.randrange(0, 2000)]
        result = least_squares(residuals, x0, bounds=bounds, max_nfev=500, verbose=1, gtol=1e-4)
        # print(result.x[0], result.x[1], result.x[2])
        success = result.cost < cost_threshold

    x = result.x
    calculated_T = r.from_euler("xyx", [x[0], x[1], x[2]], degrees=True).as_matrix()
    print("Angles for T: ", x[0], x[1], x[2])
    print("Angles for first row of F: ", x[3], x[4])
    # print("\nT: \n", T)
    # print("\nCalculated T: \n", calculated_T)
    print("cost: ", result.cost)


