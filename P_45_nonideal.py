import numpy as np
from scipy.spatial.transform import Rotation as r
from nonideal import rotation_nonideal_axes
import random

"""Calculates the angles to set the last 3 wave plates to in order to reverse F. Uses nonideal axes of rotation"""

theta = random.uniform(0, 2*np.pi)
phi = random.uniform(0, np.pi)

f = np.asarray([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])
print("f: ", f)

axes = np.asarray([[[.999633], [.0038151], [-.0268291]],
                      [[.0271085], [.999295], [.0259618]],
                      [[.9994289], [-.0335444], [.004005751]],
                      [[0], [1], [0]],
                      [[.997268], [-0.0702493], [0.0228234]],
                      [[-0.00005461419], [.999687], [-0.0250044]]])

A = np.reshape(axes[1], 3)      # axis 3
B = np.reshape(axes[2], 3)      # axis 4
d = np.cross(A, B)
det = A[0]*B[1] - A[1]*B[0]
dot = np.dot(B, f)
x = (A[0]*B[1] - A[1]*dot)/det
y = (-A[0]*B[0] + A[0]*dot)/det

a = np.sum(np.square(d))
b = 2*x*d[0] + 2*y*d[1]
c = x**2 + y**2 - 1

t = [(-b + np.sqrt(b**2 - 4*a*c))/(2*a), (-b - np.sqrt(b**2 - 4*a*c))/(2*a)]

pt = np.array([x, y, 0])

# We have two possible points for f prime. We will choose the one with the positive z component
f_prime = pt + t[1]*d
if f_prime[2] < 0:
    f_prime = pt + t[0] * d

print("f_prime: ", f_prime)

point_B = dot*B/np.sum(np.square(B))

point_A = A[0]*A/np.sum(np.square(A))

theta4 = -np.arccos(np.dot(f - point_B, f_prime - point_B)/(np.linalg.norm(f - point_B)*np.linalg.norm(f_prime - point_B)))*180/np.pi % 360

# Since we pick f_prime to have a pos z component, theta3 will always be in [180, 360]
theta3 = -np.arccos(np.dot(f_prime - point_A, np.array([1, 0, 0]) - point_A)/(np.linalg.norm(f_prime - point_A)*np.linalg.norm(np.array([1, 0, 0]) - point_A)))*180/np.pi % 360

# get the complement of theta4 if necessary
y_comp = (f - point_B) - (np.dot(f_prime - point_B, f) / np.dot(f_prime - point_B, f_prime - point_B)) * (f_prime - point_B)
if y_comp[1] < 0:
    theta4 = 360 - theta4

R3 = r.as_matrix(r.from_rotvec(A*theta3/np.linalg.norm(A), degrees=True))
R4 = r.as_matrix(r.from_rotvec(B*theta4/np.linalg.norm(B), degrees=True))

print("theta4, theta3:", theta4, theta3)

print(f @ R4)
print(f @ R4 @ R3)
