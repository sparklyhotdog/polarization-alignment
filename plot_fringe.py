from client import Client
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as r
import yaml

# Plots the measured power while rotating the 4th wave plate in the polarization analyzer for the H, V, D, A states

y_fn = open('serverinfo.yaml')
dicty = yaml.load(y_fn, Loader=yaml.SafeLoader)
y_fn.close()

analyzer = Client(dicty['PA1000']['host'], dicty['PA1000']['port'])
synth = Client(dicty['PSY201']['host'], dicty['PSY201']['port'])
pow_meter = Client(dicty['PM400']['host'], dicty['PM400']['port'])

# Set up the angles for the waveplates

P = np.zeros(6)

# Set the angles for the first 3 waveplates to emulate inverse T.
# The xyx euler angles of T are chosen from the txt output file from the least squares fitting
leastsquares_output = np.loadtxt('data/leastsquares_output.txt')
thetas = leastsquares_output[0][0:3]
P[0:3] = 360*np.ones(3) - thetas

# Set the offsets to undo F (taken from the NuCrypt config info)
P[3:6] = 360*np.ones(3) - np.asarray([117.972, 14.7918, 150.632])
P = np.round(P, 5)
print("Offsets for the waveplates: ", P)

for i in range(6):
    msg = 'retard ' + str(i) + ' ' + str(P[i])
    # print(msg)
    analyzer._send(msg)

# Start taking measurements
bases = ['h', 'v', 'd', 'a']
readings = [[], [], [], []]
n = 15
x = np.linspace(0, 360, n)
for i in range(4):
    synth._send(bases[i])       # Set polarization synthesizer to H, V, D, A

    for angle_offset in x:
        msg = 'retard 3 ' + str((P[3] + round(angle_offset, 5)) % 360)      # Change the angle of the fourth plate
        print(msg)
        print(analyzer._send(msg))
        readings[i].append(1000*float(pow_meter._send('pow?')))             # Record the power measurement

analyzer.disconnect()
synth.disconnect()
pow_meter.disconnect()

fig, ax = plt.subplots()
for i in range(4):
    ax.plot(x, readings[i], label=bases[i])

ax.set(xlabel='Theta for Waveplate 4 (deg)', ylabel='Power (mW)')
legend = plt.legend()
ax.grid()
plt.savefig("plots/fig1_nideal.png")

plt.show()
