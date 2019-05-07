import matplotlib.pyplot as plt
import numpy as np
import config

accelerometer = config.TRAININGFILES + '\\trainAccelerometer.npy'
accelerometer_data = np.load(accelerometer)
# print(accelerometer_data.shape)  # -> [N,T,S] (1692, 800, 3)

plot_accelerometer_data = True
plot_labels = False

if plot_accelerometer_data:
	t = np.arange(0., 4000., 4000 / accelerometer_data.shape[1])

	plt.title('Sequence of first activity, Throw out', fontsize=16)
	plt.xlabel('Time in ms', fontsize=14, color='blue')
	plt.ylabel('Values', fontsize=14, color='blue')
	plt.plot(t, accelerometer_data[0][:, 0], 'b--', t, accelerometer_data[0][:, 1], 'g--', t, accelerometer_data[0][:, 2], 'r--')
	plt.show()

	plt.title('Sequence of second activity, Open lid by rotate', fontsize=16)
	plt.xlabel('Time in ms', fontsize=14, color='blue')
	plt.ylabel('Values', fontsize=14, color='blue')
	plt.plot(t, accelerometer_data[1][:, 0], 'b--', t, accelerometer_data[1][:, 1], 'g--', t, accelerometer_data[1][:, 2], 'r--')
	plt.show()

	plt.title('Sequence of tenth activity, Lying down', fontsize=16)
	plt.xlabel('Time in ms', fontsize=14, color='blue')
	plt.ylabel('Values', fontsize=14, color='blue')
	plt.plot(t, accelerometer_data[9][:, 0], 'b--', t, accelerometer_data[9][:, 1], 'g--', t, accelerometer_data[9][:, 2], 'r--')
	plt.show()

labels = config.TRAININGFILES + '\\trainLabels.npy'
labels_data = np.load(labels)
print(labels_data.shape)  # -> (1692,)
print(labels_data[:10])  # -> [47 22 45 13 17 42 29 34  8 17]
