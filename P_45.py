import numpy as np
from scipy.spatial.transform import Rotation as r

"""Calculates the angles to set the last 3 wave plates to in order to reverse F. Uses ideal axes of rotation"""

theta = 1.74850349
phi = 0.62496112

f = np.asarray([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])
print("f: ", f)

theta3 = 360 - np.arccos(f[0])*180/np.pi
theta4 = np.arccos(f[2]/np.sqrt(f[1]**2 + f[2]**2))*180/np.pi
if f[1] > 0:
    theta4 = 360 - theta4
print("theta3, theta4: ", theta3, theta4)
R4 = r.as_matrix(r.from_rotvec([theta4, 0, 0], degrees=True))       # x (H)
R3 = r.as_matrix(r.from_rotvec([0, theta3, 0], degrees=True))       # y (D)

# print("f R4: ", f@R4)

print(np.round(f @ R4 @ R3, 4))     # Set R5 as the identity matrix

