import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as r
import random


def r2x(v):
    """Returns a scipy rotation that maps vector v to the x vector"""
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
    """Returns a rotation matrix that rotates theta (radians) around axis"""
    rot_vec = np.resize(axis * theta / np.linalg.norm(axis), 3)
    return r.as_matrix(r.from_rotvec(rot_vec))


def rot_prime(axis, theta):
    """ Returns the differential of the rotation matrix rot(axis, theta) with respect to theta"""
    R_x = np.asarray([[1, 0, 0],
                      [0, -np.sin(theta), -np.cos(theta)],
                      [0, np.cos(theta), -np.sin(theta)]])
    A2X = r2x(axis)
    return np.transpose(r.as_matrix(A2X)) @ R_x @ r.as_matrix(A2X)


def rotation_nonideal_axes(axes: npt.NDArray, rot_angles: npt.ArrayLike, degrees: bool = False) -> npt.NDArray:
    """ Returns the rotation matrix of a rotation in terms of its rotation axes and rotation angles.
    axes should be a nx3 array. rot_angles should be a n array"""
    # TODO: also be able to accommodate a list of rotations
    rotation = r.identity()
    for i in range(axes.shape[0]):
        R_i = r.from_rotvec(np.reshape(axes[i], 3) * rot_angles[i] / np.linalg.norm(axes[i]), degrees=degrees)
        rotation = R_i * rotation

    return r.as_matrix(rotation)    # TODO: change return type to scipy rotation?


def calculate_euler_angles(M_goal, seq, axes, error_threshold=1e-10, verbose=False, degrees=False):
    """Returns the retardance angles needed to achieve rotation matrix M_goal
    given non-ideal axes of rotation. Based on Nucrypt pa1000_math_docver2.docx

    M_goal is the goal rotation matrix. seq should be hdh/xyx or dhd/yxy. axes should be a 3x3 array of the rotation
    axes. Setting verbose to true will print the angles and error each iteration."""

    if seq == 'hdh':
        seq = 'xyx'
    if seq == 'dhd':
        seq = 'yxy'

    # trim M_goal if needed
    if M_goal.shape == (4, 4):
        M_goal = np.delete(M_goal, 0, 0)
        M_goal = np.delete(M_goal, 0, 1)
    R_goal = r.from_matrix(M_goal)

    # initial guess with the ideal case
    # use "xyx" for HDH; "yxy" for DHD
    P = r.as_euler(R_goal, seq)
    error = 1e20

    for i in range(100):
        M_est = rotation_nonideal_axes(axes, P)
        prev_error = error
        error = np.linalg.norm(M_est - M_goal)
        if verbose:
            print("Error: ", error)
            print("Angles (deg): ", P*180/np.pi)

        if error < error_threshold:
            if verbose:
                print("Error threshold reached")
                print(M_goal - M_est)
            if degrees:
                return P * 180 / np.pi % 360
            return P
        if error >= prev_error:
            if verbose:
                print("Error Diverging")
            if degrees:
                return P * 180 / np.pi % 360
            return P

        # Compute the 9x3 Jacobian
        J1 = rot(axes[2], P[2]) @ rot(axes[1], P[1]) @ rot_prime(axes[0], P[0])
        J2 = rot(axes[2], P[2]) @ rot_prime(axes[1], P[1]) @ rot(axes[0], P[0])
        J3 = rot_prime(axes[2], P[2]) @ rot(axes[1], P[1]) @ rot(axes[0], P[0])
        J1 = np.reshape(J1, (9, 1))
        J2 = np.reshape(J2, (9, 1))
        J3 = np.reshape(J3, (9, 1))

        J = np.hstack((J1, J2, J3))
        JTJ = np.transpose(J) @ J

        if np.linalg.det(JTJ) == 0:
            # JTJ is not invertible. This means that there is not one unique solution, but multiple. We choose to hold
            # the third waveplate still, and just change the first one.
            if verbose:
                print("Gimbal Lock")
            J = np.delete(J, 2, axis=1)

        # M_est + J @ delta P = M_goal
        # J @ delta P = (M_goal - M_est)
        # We have 9 equations, and 3 unknowns, so we use the pseudo inverse of J to calculate delta P
        delta_P = np.linalg.pinv(J) @ (np.reshape(M_goal, (9, 1)) - np.reshape(M_est, (9, 1)))

        P = P + np.reshape(delta_P, 3)

    if degrees:
        return P * 180 / np.pi % 360
    return P


def calc_ret_angles_for_F_trig(F, axes):
    """Returns the angles (degrees) to set the last 3 variable retarders in order to reverse the F transformation,
    given the first row of F (an array length 3), and the rotation axes (a 6x3 array, column vectors are fine).

    The output is [theta3, theta4], where theta3 is the angle to set the fourth variable retarder,
    theta4 is the angle to set the fifth variable retarder (indexing starting at 0),
    and the last variable retarder is set to 0.

    Assumes that the fourth and fifth variable retarders are D and H respectively."""

    A = np.reshape(axes[3], 3)  # axis 3
    B = np.reshape(axes[4], 3)  # axis 4
    d = np.cross(A, B)
    det = A[0] * B[1] - A[1] * B[0]
    dot = np.dot(B, F)
    x = (A[0] * B[1] - A[1] * dot) / det
    y = (-A[0] * B[0] + A[0] * dot) / det

    a = np.sum(np.square(d))
    b = 2 * x * d[0] + 2 * y * d[1]
    c = x ** 2 + y ** 2 - 1

    t = [(-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a), (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)]

    pt = np.array([x, y, 0])

    # We have two possible points for f prime. We will choose the one with the positive z component.
    # (This part assumes that we have D-H-D aligned retarders for the last 3)
    f_prime = pt + t[1] * d
    if f_prime[2] < 0:
        f_prime = pt + t[0] * d

    point_B = dot * B / np.sum(np.square(B))

    point_A = A[0] * A / np.sum(np.square(A))

    theta4 = -np.arccos(np.dot(F - point_B, f_prime - point_B) / (
            np.linalg.norm(F - point_B) * np.linalg.norm(f_prime - point_B))) * 180 / np.pi % 360

    # Since we pick f_prime to have a pos z component, theta3 will always be in [180, 360]
    theta3 = -np.arccos(np.dot(f_prime - point_A, np.array([1, 0, 0]) - point_A) / (
            np.linalg.norm(f_prime - point_A) * np.linalg.norm(np.array([1, 0, 0]) - point_A))) * 180 / np.pi % 360

    # Get the complement of theta4 if necessary
    y_comp = (F - point_B) - (np.dot(f_prime - point_B, F) / np.dot(f_prime - point_B, f_prime - point_B)) * (
            f_prime - point_B)
    if y_comp[1] < 0:
        theta4 = 360 - theta4

    return [theta3, theta4]

if __name__ == "__main__":
    T = r.as_matrix(r.random())
    nonideal_axes = np.asarray([[[.999633], [.0038151], [-.0268291]],
                                [[.0271085], [.999295], [.0259618]],
                                [[.9994289], [-.0335444], [.004005751]],
                                [[0], [1], [0]],
                                [[.997268], [-0.0702493], [0.0228234]],
                                [[-0.00005461419], [.999687], [-0.0250044]]])

    angles = calculate_euler_angles(T, "xyx", nonideal_axes[0:3], verbose=True)
    print("Angles (rad): ", angles)

    print(T)
    print(rotation_nonideal_axes(nonideal_axes[0:3], angles))

