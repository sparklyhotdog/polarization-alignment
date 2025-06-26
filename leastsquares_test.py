import unittest
import numpy as np
from scipy.spatial.transform import Rotation as r
import random
from leastsquares_fiberized import generate_expected_counts, least_squares_fitting, calc_ret_angles_from_matrix, calc_ret_angles_from_x, cost
from nonideal import rotation_nonideal_axes

nonideal_axes = np.asarray([[[.999633], [.0038151], [-.0268291]],
                                [[.0271085], [.999295], [.0259618]],
                                [[.9994289], [-.0335444], [.004005751]],
                                [[0], [1], [0]],
                                [[.997268], [-0.0702493], [0.0228234]],
                                [[-0.00005461419], [.999687], [-0.0250044]]])
ideal_axes = np.asarray([[[1], [0], [0]], [[0], [1], [0]],
                         [[1], [0], [0]], [[0], [1], [0]],
                         [[1], [0], [0]], [[0], [1], [0]]])
Hpol_det = np.asmatrix(np.asarray([[0.5, 0.5, 0, 0]]))
rotations = r.from_rotvec(np.asarray([[0, 0, 0],
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


def mueller(M):
    """Returns the Mueller matrix equivalent of a 3x3 matrix M"""
    matrix = np.zeros((4, 4))
    matrix[1:4, 1:4] = M
    matrix[0][0] = 1
    return matrix


def is_rot_matrix(M, error_threshold=1e-5):
    return abs(np.linalg.det(M) - 1) < error_threshold and np.linalg.norm(np.transpose(M) - np.linalg.inv(M)) < error_threshold


class LeastSquaresTest(unittest.TestCase):
    error_threshold = 1e-5

    def setUp(self):
        self.T = r.as_matrix(r.random())
        self.F = r.as_matrix(r.random())

    # Testing calc_ret_angles_from_matrix -------------------------------------------------------------

    def testP_reverse(self):
        angles = calc_ret_angles_from_matrix(self.T, self.F[0], ideal_axes)
        P_matrix = rotation_nonideal_axes(ideal_axes, angles, degrees=True)

        transformation = Hpol_det @ mueller(self.F) @ mueller(P_matrix) @ mueller(self.T)
        print("should be [0.5 0.5 0 0]: ", transformation)

        self.assertTrue(np.linalg.norm(Hpol_det - transformation) < LeastSquaresTest.error_threshold)

    def testP_reverse_nonideal(self):
        angles = calc_ret_angles_from_matrix(self.T, self.F[0], nonideal_axes)
        P_matrix = rotation_nonideal_axes(nonideal_axes, angles, degrees=True)

        transformation = Hpol_det @ mueller(self.F) @ mueller(P_matrix) @ mueller(self.T)
        print("should be [0.5 0.5 0 0]: ", transformation)

        self.assertTrue(np.linalg.norm(Hpol_det - transformation) < LeastSquaresTest.error_threshold)

    def testT_reverse_nonideal(self):
        angles = calc_ret_angles_from_matrix(self.T, self.F[0], nonideal_axes)

        P_210 = rotation_nonideal_axes(nonideal_axes[0:3], angles[0:3], degrees=True)
        print("should be identity: ", self.T @ P_210)       # should be identity matrix

        self.assertTrue(np.linalg.norm(self.T @ P_210 - np.identity(3)) < LeastSquaresTest.error_threshold)

    def test_fitting(self):
        counts, actual_x = generate_expected_counts(rotations, sigma=.002)
        result = least_squares_fitting(counts, rotations, axes=ideal_axes)

        x = result.x
        calculated_T = rotation_nonideal_axes(ideal_axes[0:3], x[0:3], degrees=True)
        calculated_F = np.asarray([np.cos(x[3]) * np.sin(x[4]), np.sin(x[3]) * np.sin(x[4]), np.cos(x[4])])
        other_T = np.copy(calculated_T)
        other_T[:, 0:2] = -calculated_T[:, 0:2]
        other_F = -calculated_F

        self.assertTrue(is_rot_matrix(calculated_T) and is_rot_matrix(other_T))
        print("\nCalculated T: \n", calculated_T)
        print("First row of F: ", calculated_F)
        print("Result: ", x)
        print("Cost: ", result.cost)
        print("cost of actual x", cost(actual_x, counts, rotations, ideal_axes))

        angles_from_x = calc_ret_angles_from_x(x)
        angles_from_matrix = calc_ret_angles_from_matrix(calculated_T, calculated_F, ideal_axes)
        print("angles from actual T and F", calc_ret_angles_from_matrix(self.T, self.F[0], ideal_axes))
        print(Hpol_det @ mueller(self.F) @ mueller(rotation_nonideal_axes(ideal_axes, calc_ret_angles_from_matrix(self.T, self.F[0], ideal_axes), degrees=True)) @ mueller(self.T))
        print(Hpol_det @ mueller(self.F) @ mueller(rotation_nonideal_axes(ideal_axes, calc_ret_angles_from_matrix(calculated_T, calculated_F, ideal_axes), degrees=True)) @ mueller(self.T))
        print(Hpol_det @ mueller(self.F) @ mueller(rotation_nonideal_axes(ideal_axes, calc_ret_angles_from_matrix(other_T, other_F, ideal_axes), degrees=True)) @ mueller(self.T))

        print("other calculated:", calc_ret_angles_from_matrix(other_T, other_F, ideal_axes))
        print(angles_from_x, angles_from_matrix)

        P_matrix = rotation_nonideal_axes(ideal_axes, angles_from_x, degrees=True)

        transformation = Hpol_det @ mueller(self.F) @ mueller(P_matrix) @ mueller(self.T)

        print(transformation)





if __name__ == '__main__':
    unittest.main()
