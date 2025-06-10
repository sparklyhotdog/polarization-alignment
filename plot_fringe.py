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

P = np.zeros(6)

# inverse T, given the xyx euler angles of T
P[0:3] = 360*np.ones(3) - np.asarray([301.47538674, 128.56914548,   9.17289652])

# these are the offsets to undo F
P[3:6] = 360*np.ones(3) - np.asarray([117.972, 14.7918, 150.632])
P = np.round(P, 5)
print(P)

for i in range(6):
    msg = 'retard ' + str(i) + ' ' + str(P[i])
    print(msg)
    # analyzer._send(msg)

bases = ['h', 'v', 'd', 'a']
readings = [[], [], [], []]
n = 15
x = np.linspace(0, 360, n)      # max for NuCrypt retarders is 420
for i in range(4):
    # Set polarization synthesizer to h, v, d, or a
    synth._send(bases[i])

    for angle_offset in x:
        msg = 'retard 3 ' + str((P[3] + round(angle_offset, 5)) % 360)
        print(msg)
        print(analyzer._send(msg))
        readings[i].append(1000*float(pow_meter._send('pow?')))

analyzer.disconnect()
synth.disconnect()
pow_meter.disconnect()

fig, ax = plt.subplots()
for i in range(4):
    ax.plot(x, readings[i], label=bases[i])

ax.set(xlabel='Theta for Waveplate 4 (deg)', ylabel='Power (mW)')
legend = plt.legend()
ax.grid()
plt.savefig("plots/fig1a.png")

plt.show()
