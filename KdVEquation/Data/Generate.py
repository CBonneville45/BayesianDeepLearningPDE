import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

np.random.seed(7)

Utrue = np.loadtxt('Utrue.txt', delimiter = ',')
Utrue = np.flip(Utrue, axis = 0)

nx, nt = Utrue.shape
x = np.linspace(-20, 20, nx)
t = np.linspace(0, 40, nt)

x, t = np.meshgrid(x, t)
x = x.T
x = np.flip(x, axis = 0)
t = t.T

index_t = np.arange(0, nt, 80)
nt = index_t.shape[0]
Utrue = Utrue[:, index_t]
t = t[:, index_t]
x = x[:, index_t]

n_sensors = 16
sensors = np.arange(0, nx, 1)
np.random.shuffle(sensors)
sensors = sensors[0:n_sensors]

X = np.zeros([1, 2])
y = np.zeros(1)

for i in range(n_sensors):

    X_i = np.hstack((t[sensors[i], :].reshape(-1, 1), np.ones([nt, 1]) * x[sensors[i], 0]))
    y_i = Utrue[sensors[i], :]

    X = np.vstack((X, X_i))
    y = np.hstack((y, y_i))

X = X[1:, :]
y = y[1:]

X, y = shuffle(X, y)

fig = plt.figure(figsize = (11, 7))

plt.contourf(t, x, Utrue, 100, cmap = plt.cm.jet)
plt.title('True')
plt.ylabel('x')
plt.xlabel('t')
plt.colorbar()
plt.scatter(X[:, 0], X[:, 1], color = 'k', marker = '.', s = 5)
plt.show()

y_000 = np.copy(y)
y_001 = np.copy(y) + np.random.randn(n_sensors * nt) * 0.01
y_005 = np.copy(y) + np.random.randn(n_sensors * nt) * 0.05

Data_000 = {'X' : X, 'y' : y_000}
Data_001 = {'X' : X, 'y' : y_001}
Data_005 = {'X' : X, 'y' : y_005}

np.save('RandomSensorsDataPointsNoise0.00.npy', Data_000)
np.save('RandomSensorsDataPointsNoise0.01.npy', Data_001)
np.save('RandomSensorsDataPointsNoise0.05.npy', Data_005)