from client import Client
import matplotlib.pyplot as plt
import numpy as np

# Plots the measured power while rotating one of the wave plates in the polarization analyzer, controlled remotely

HOST = '169.254.142.114'
PA_port = 56000
PSY_port = 56010
PM_port = 56001

analyzer = Client(HOST, PA_port)
synth = Client(HOST, PSY_port)
pow_meter = Client(HOST, PM_port)

analyzer._send('retard 0 270')
analyzer._send('retard 1 57.3')
analyzer._send('retard 2 0')

readings = []
n = 15
x = np.linspace(0, 420, n)
for theta in x:
    msg = 'retard 2 ' + str(theta)
    # print(msg)
    print(analyzer._send(msg))
    # print(analyzer._send('retardances?'))
    readings.append(1000*float(pow_meter._send('pow?')))

print("\nPower readings: ", readings)
analyzer.disconnect()
pow_meter.disconnect()

fig, ax = plt.subplots()
ax.plot(x, readings)

ax.set(xlabel='Theta for Waveplate 3 (deg)', ylabel='Power (mW)', title='[270, 57.3, x, 0, 0, 0]')
ax.grid()
plt.savefig("plots/270-57.3-x-0-0-0.png")

plt.show()
