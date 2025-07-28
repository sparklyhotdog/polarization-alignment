import numpy as np
from scipy.spatial.transform import Rotation as r
from scipy.optimize import least_squares, direct, minimize
import random, time
from measurements import measure_HD, measure_HD_fast
from nonideal import rotation_nonideal_axes, calculate_euler_angles, calc_ret_angles_for_F_trig
from plot_fringe import plot, plot2, plot_scatter


class Fiberized:
    # The 6 axes of rotation for the H-D-H-D-H-D variable retarders in the Nucrypt PA (taken from the flash of the PA).
    # The 4th variable retarder is the reference
    axes = np.asarray([[.999633, .0038151, -.0268291],
                      [.0271085, .999295, .0259618],
                      [.9994289, -.0335444, .004005751],
                      [0, 1, 0],
                      [.997268, -0.0702493, 0.0228234],
                      [-0.00005461419, .999687, -0.0250044]])

    def __init__(self, num_rand_rots=10, rotation_list=None, verbose=True):
        """ Performs the reference frame alignment and calculates the T and F matrices,
        as well as the retardance angles to set to reverse them.

        For the rotations: by default, it will choose random angles to create random rotations.

        num_rand_rots is the number of random rotations it will take measurements for. Should be at least 8.

        Set rotation_list to a scipy rotation object containing multiple rotations.
        Don't set this unless you want particular rotations for the measurements.
        This will be slower than using random rotations.

        verbose, if true, will print each message sent and received by the polarization analyzer"""

        start_time = time.time()
        if rotation_list is not None:
            self.counts, angles = measure_HD(r.as_euler(rotation_list, "xyx", degrees=True), verbose=verbose)
            num_rots = len(rotation_list)
        else:
            num_rots = num_rand_rots
            self.counts, angles = measure_HD_fast(num_rots=num_rots, verbose=verbose)


        # Update the rotation list based on the angles given back by the polarization analyzer
        self.rotations = r.identity(num_rots)
        for i in range(num_rots):
            self.rotations[i] = r.from_matrix(rotation_nonideal_axes(Fiberized.axes, angles[i], degrees=True))

        # The least squares solves for x [theta1, theta2, theta3, theta, phi, N_H, N_D], where
        # theta1, theta2, theta3 [0, 360] are the angles around the rotation axes (degrees),
        # theta [0, 2pi] and phi [0, pi] are the spherical angles for the first row of the F matrix (radians),
        # and N_H and N_D are the max power for the H and D states from the synthesizer (mW)
        leastsquares_start = time.time()
        self.least_squares_result = least_squares_fitting(self.counts, self.rotations, axes=Fiberized.axes)
        x = self.least_squares_result.x

        self.T = rotation_nonideal_axes(Fiberized.axes[0:3], x[0:3], degrees=True)
        self.F = np.asarray([np.cos(x[3]) * np.sin(x[4]), np.sin(x[3]) * np.sin(x[4]), np.cos(x[4])])
        self.N_H = x[5]
        self.N_D = x[6]

        # TODO: find the exact value of this? prob not exact because of nonideal axes
        self.other_T = np.copy(self.T)
        self.other_T[:, 0:2] = -self.T[:, 0:2]
        self.other_F = -self.F

        self.ret_angles = np.empty(6)
        self.ret_angles[0:3] = calculate_euler_angles(np.linalg.inv(self.T), 'hdh', self.axes[0:3], degrees=True)
        self.ret_angles[3:6] = calculate_euler_angles(np.linalg.inv(extend_F_from_first_row(self.F)), 'dhd', self.axes[3:6], degrees=True)

        self.other_ret_angles = np.empty(6)
        self.other_ret_angles[0:3] = calculate_euler_angles(np.linalg.inv(self.other_T), 'hdh', self.axes[0:3], degrees=True)
        self.other_ret_angles[3:6] = calculate_euler_angles(np.linalg.inv(extend_F_from_first_row(self.other_F)), 'dhd', self.axes[3:6], degrees=True)

        # Assuming that F does not change and the correct ret angles are close to [380 160 0]
        if (self.ret_angles[3] - 280)**2 + (self.ret_angles[4] - 160)**2 > (self.other_ret_angles[3] - 280)**2 + (self.other_ret_angles[4] - 160)**2:
            # The correct one is the second one and we switch
            tempT = self.other_T
            tempF = self.other_F
            temp_ret_angles = self.other_ret_angles
            self.other_T = self.T
            self.other_F = self.F
            self.other_ret_angles = self.ret_angles
            self.T = tempT
            self.F = tempF
            self.ret_angles = temp_ret_angles

        end_time = time.time()
        self.duration = end_time - start_time
        self.calc_time = end_time - leastsquares_start

    def print_results(self):
        """Prints the calculated T and F matrices, the count rates of the H and D states, the cost of the fitting,
        the calculated retardance angles to reverse T and F, and the time it took"""
        print("Calculated T: \n", self.T)
        print("Calculated first row of F: \n", self.F)
        print("N_H, N_D: ", self.N_H, self.N_D)
        print("Cost: ", self.least_squares_result.cost)
        print("Retardance angles: \n", self.ret_angles, '\n', self.other_ret_angles)
        print("Total time taken (s): ", np.round(self.duration, 2))
        print("Calculation time (s): ", np.round(self.calc_time, 2))

    def plot_fringes(self, filepath=None, verbose=False, num_points=10):
        """Plots the fringes with compensation--both T and the other T in subplots."""
        angles_str = str(np.round(self.ret_angles, 2)) + '\n' + str(np.round(self.other_ret_angles, 2))
        plot2(ret_angles=[self.ret_angles, self.other_ret_angles], title=angles_str, filepath=filepath, verbose=verbose, num_points=num_points)

    def plot_fringe(self, filepath=None, verbose=False, num_points=10):
        """Plots the fringes with compensation (from the calculated T and F)"""
        title = "With compensation\n" + str(np.round(self.ret_angles, 2))
        plot(ret_angles=self.ret_angles, title=title, filepath=filepath, verbose=verbose, num_points=num_points)

    def plot_scatter(self, filepath=None, verbose=False, num_points=10):
        """Plots the fringes with compensation as a scatter plot (from the calculated T and F)
        with the expected sinusoids as dashed lines"""
        title = "With compensation"
        plot_scatter((self.N_H + self.N_D)/2, ret_angles=self.ret_angles, title=title, filepath=filepath, verbose=verbose, num_points=num_points)

    def plot_fringe_other(self, filepath=None, verbose=False, num_points=10):
        """Plots the fringes with compensation (from the other T and F)"""
        title = "With compensation \n" + str(np.round(self.ret_angles, 2))
        plot(ret_angles=self.other_ret_angles, title=title, filepath=filepath, verbose=verbose, num_points=num_points)


def residuals(var, count_data, rotation_list, axes=None):
    """"Returns the residuals for each measurement, given an array of our unknowns, the count measurements, a scipy
    rotation object containing the rotations, and an optional array of non-ideal axes of rotation (nx3 array).

    var is an array representing [theta1, theta2, theta3, theta, phi, N_H, N_D], where
    theta1, theta2, theta3 [0, 360] are the euler angles (xyx, or around the nonideal axes if provided) in degrees,
    theta [0, 2pi] and phi [0, pi] are the spherical angles for the first row of F,
    N_H and N_D are the max power for the H and D states from the synthesizer (mW)"""
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

        # Map C_H to the even indices, and C_D to the odd indices
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
                     (max_C_D, 1.5 * max_C_D)]      # the direct method uses a different bounds formatting

    # first calculate an initial guess using a global optimization algorithm
    initial_result = direct(cost, bounds_direct, args=(count_data, rotation_list, axes))
    x0 = initial_result.x
    # then use the least-squares algorithm with our initial guess to find a more exact solution
    fitting = least_squares(residuals, x0, bounds=bounds, max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-10, verbose=verbose, args=(count_data, rotation_list, axes))
    return fitting


def extend_F_from_first_row(F_row1):
    """F_row1 should be an array length 3. Returns a rotation matrix that has F_row1 as its first row."""
    # We choose the second row such that the x component is 0
    F_row2 = np.array([0, F_row1[2], -F_row1[1]])
    F_row2 = F_row2 / np.linalg.norm(F_row2)
    # The third row must be orthogonal to both the first and second, so we set it to be the cross product.
    F_row3 = np.cross(F_row1, F_row2)

    return np.array([F_row1, F_row2, F_row3])


if __name__ == "__main__":
    A = Fiberized(num_rand_rots=10, verbose=False)
    # A = Fiberized(rotation_list=r.random(16), verbose=False)
    A.print_results()
    # choose angle set that is closest to 280 160 for F
    path = 'plots/jul28__4.png'
    # A.plot_fringe(filepath=path, verbose=False, num_points=10)
    A.plot_scatter(filepath=path, verbose=False, num_points=10)
    plot_scatter((A.N_H + A.N_D)/2, title='No compensation', filepath='plots/jul28_nocompensation_4.png', verbose=True, num_points=10)

