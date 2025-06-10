"""
Created on Fri May 23 12:18:15 2025
"""

import numpy as np
from scipy.spatial.transform import Rotation as r
import random


def r2x(v):
    """Returns a rotation matrix that maps vector v to the x vector"""
    v = np.reshape(v, 3)
    x = [1, 0, 0]

    # axis of rotation is v cross x
    axis = np.cross(v, x)

    # angle of rotation calculated by the v dot x
    theta = np.arccos(np.dot(v, x)/(np.linalg.norm(v)*np.linalg.norm(x)))

    rot_vec = axis * theta/np.linalg.norm(axis)
    # ensures length of rot_vec is equal to theta

    return r.from_rotvec(rot_vec)


def rot(axis, theta):
    """Returns a rotation matrix that rotates theta around axis"""
    rot_vec = np.resize(axis * theta / np.linalg.norm(axis), 3)
    return r.as_matrix(r.from_rotvec(rot_vec))


def rot_prime(axis, theta):
    """ Returns the differential of the rotation matrix rot(axis, theta) with respect to theta"""
    R_x = np.asarray([[1, 0, 0],
                      [0, -np.sin(theta), -np.cos(theta)],
                      [0, np.cos(theta), -np.sin(theta)]])
    A2X = r2x(axis)
    return np.transpose(r.as_matrix(A2X)) @ R_x @ r.as_matrix(A2X)


def rotation_nonideal_axes(A1, A2, A3, P, degrees=False):
    """ Returns the rotation matrix of the series of rotations around axes A1, A2, and A3 with angles corresponding to
    the 3d vector P """
    R1 = r.from_rotvec(np.reshape(A1, 3) * P[0] / np.linalg.norm(A1), degrees=degrees)
    R2 = r.from_rotvec(np.reshape(A2, 3) * P[1] / np.linalg.norm(A2), degrees=degrees)
    R3 = r.from_rotvec(np.reshape(A3, 3) * P[2] / np.linalg.norm(A3), degrees=degrees)

    return r.as_matrix(R3 * R2 * R1)


def calculate_euler_angles(A1, A2, A3, M_goal):
    """Calculates the retardance angles needed to achieve rotation matrix M_goal
    given non-ideal axes of rotation A1, A2, A3"""
    error_threshold = .01
    # trim M_goal
    if M_goal.shape == (4,4):
        M_goal = np.delete(M_goal, 0, 0)
        M_goal = np.delete(M_goal, 0, 1)
    R_goal = r.from_matrix(M_goal)

    # initial guess with the ideal case
    # use "xyx" for HDH; "yxy" for DHD
    P = r.as_euler(R_goal, "xyx")
    prev_error = 1e20

    for i in range(10):
        # print(P)
        M_est = rotation_nonideal_axes(A1, A2, A3, P)
        error = np.linalg.norm(M_est - M_goal)
        print("Error: ", error)

        if error < error_threshold:
            print("Error threshold reached")
            print(M_goal - M_est)
            return P
        if error >= prev_error:
            # error diverging
            print("Error Diverging")
            return P

        # Compute the 9x3 Jacobian
        J1 = rot(A3, P[2]) @ rot(A2, P[1]) @ rot_prime(A1, P[0])
        J2 = rot(A3, P[2]) @ rot_prime(A2, P[1]) @ rot(A1, P[0])
        J3 = rot_prime(A3, P[2]) @ rot(A2, P[1]) @ rot(A1, P[0])
        J1 = np.reshape(J1, (9, 1))
        J2 = np.reshape(J2, (9, 1))
        J3 = np.reshape(J3, (9, 1))

        J = np.hstack((J1, J2, J3))
        JTJ = np.transpose(J) @ J

        if np.linalg.det(JTJ) == 0:
            # JTJ is not invertible. This means that there is not one unique solution, but multiple. We choose to hold
            # the third waveplate still, and just change the first one.
            J = np.delete(J, 2, axis=1)

        # M_est + J @ delta P = M_goal
        # J @ delta P = (M_goal - M_est)
        # We have 9 equations, and 3 unknowns, so we use the pseudo inverse of J to calculate delta P
        delta_P = np.linalg.pinv(J) @ (np.reshape(M_goal, (9, 1)) - np.reshape(M_est, (9, 1)))

        P = P + np.reshape(delta_P, 3)

    return P


if __name__ == "__main__":
    T = r.as_matrix(r.from_euler("xyz", [2*np.pi*random.random(), 2*np.pi*random.random(), 2*np.pi*random.random()]))
    axis1 = np.asarray([[.999633], [.0038151], [-.0268291]])
    axis2 = np.asarray([[.0271085], [.999295], [.0259618]])
    axis3 = np.asarray([[.9994289], [-.0335444], [.004005751]])

    angles = calculate_euler_angles(axis1, axis2, axis3, T)
    print("\nAngles (rad): ", angles)

