"""
Created on Wed May 21 15:17:44 2025
"""

import numpy as np
import random

H_state = np.matrix([[1], [1], [0], [0]])
D_state = np.matrix([[1], [0], [1], [0]])

"""
H_ret/D_ret(theta): constructing the muller matrix of a H-waveplate and a D-waveplate ("H/D retarder");
see also page 4 of nucrypt math cover;
"""


def H_ret(theta):
    mat = np.matrix(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.cos(theta), -1 * np.sin(theta)], [0, 0, np.sin(theta), np.cos(theta)]])
    return mat


def D_ret(theta):
    mat = np.matrix(
        [[1, 0, 0, 0], [0, np.cos(theta), 0, np.sin(theta)], [0, 0, 1, 0], [0, -1 * np.sin(theta), 0, np.cos(theta)]])
    return mat


"""
HDH/DHD(theta) are simply emulating the muller matrix of a H-D-H/D-H-D waveplate combination.
Here theta is a 3 element array [theta0, theta1, theta2], since you have 3 waveplates now.
"""


def HDH(theta):
    mat = H_ret(theta[2]) * D_ret(theta[1]) * H_ret(theta[0])
    return mat


def DHD(theta):
    mat = D_ret(theta[2]) * H_ret(theta[1]) * D_ret(theta[0])
    return mat


"""
a general 3D rotational matrix
definition follows this wikipedia page: https://en.wikipedia.org/wiki/Rotation_matrix;

There are several different ways doing this, I am using the yaw-pitch-roll convention.
"""


def R_general(r):
    alpha = r[0]
    beta = r[1]
    gamma = r[2]

    # yaw
    Rz = np.matrix(
        [[1, 0, 0, 0], [0, np.cos(alpha), -1 * np.sin(alpha), 0], [0, np.sin(alpha), np.cos(alpha), 0], [0, 0, 0, 1]])

    # pitch
    Ry = np.matrix(
        [[1, 0, 0, 0], [0, np.cos(beta), 0, np.sin(beta)], [0, 0, 1, 0], [0, -1 * np.sin(beta), 0, np.cos(beta)]])

    # roll
    Rx = np.matrix(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.cos(gamma), -1 * np.sin(gamma)], [0, 0, np.sin(gamma), np.cos(gamma)]])

    mat = Rz * Ry * Rx
    return mat


"""
Given an arbitrary rotation matrix T,
We calculate theta1, theta2, theta3, such that HDH/DHD(theta1, theta2, theta3) = T
Assuming the axes of rotation of the wave plates are ideal

t is a 4x4 Mueller matrix.
"""


def euler_angles(T, is_hdh):

    if is_hdh:
        n1 = H_state[1:]
        n2 = D_state[1:]
        n3 = H_state[1:]
    else:
        n1 = D_state[1:]
        n2 = H_state[1:]
        n3 = D_state[1:]

    n1T = np.transpose(n1)
    n2T = np.transpose(n2)
    n3T = np.transpose(n3)

    # A is the 3x3 trimmed version of T
    A = np.delete(T, 0, 0)
    A = np.delete(A, 0, 1)

    # Calculate thetas (from https://www.malcolmdshuster.com/Pub_2003d_J_GenEuler_MDS.pdf)
    theta1 = -np.arctan2(n3T * A * n2, -(n3T * A * np.transpose(np.cross(n1T, n2T))))[0, 0]
    theta2 = -np.arccos(n3T * A * n1)[0, 0]
    theta3 = -np.arctan2(n2T * A * n1, -(np.cross(n2T, n3T) * A * n1))[0, 0]

    return np.array([theta1, theta2, theta3])


if __name__ == "__main__":
    # Choose arbitrary x, y, and z rotation angles for T
    a = 2 * np.pi * random.random()
    b = 2 * np.pi * random.random()
    c = 2 * np.pi * random.random()

    T = R_general(np.array([a, b, c]))

    theta = euler_angles(T, True)

    print("T:\n", T)
    print("\nTheta: ", theta)
    print("\n|| T - HDH(theta) || = ", np.linalg.norm(T - HDH(theta)))
    print("\n|| T - DHD(theta) || = ", np.linalg.norm(T - DHD(theta)))



