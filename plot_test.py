from client import Client
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Plots the measured power while rotating one of the wave plates in the polarization analyzer, controlled remotely

y_fn = open('serverinfo.yaml')
dicty = yaml.load(y_fn, Loader=yaml.SafeLoader)
y_fn.close()

analyzer = Client(dicty['PA1000']['host'], dicty['PA1000']['port'])
synth = Client(dicty['PSY201']['host'], dicty['PSY201']['port'])
pow_meter = Client(dicty['PM400']['host'], dicty['PM400']['port'])

analyzer._send('retard 0 270')
analyzer._send('retard 1 57.3')
analyzer._send('retard 2 0')

readings = []
n = 15
x = np.linspace(0, 360, n)      # max for NuCrypt retarders is 420
for theta in x:
    msg = 'retard 3 ' + str(round(theta, 5))
    print(msg)
    print(analyzer._send(msg))
    # print(analyzer._send('retardances?'))
    readings.append(1000*float(pow_meter._send('pow?')))

print("\nPower readings: ", readings)
analyzer.disconnect()
pow_meter.disconnect()

fig, ax = plt.subplots()
ax.plot(x, readings)

ax.set(xlabel='Theta for Waveplate 4 (deg)', ylabel='Power (mW)', title='[270, 57.3, 0, x, 0, 0]')
ax.grid()
plt.savefig("plots/270-57.3-0-x-0-0.png")

plt.show()
