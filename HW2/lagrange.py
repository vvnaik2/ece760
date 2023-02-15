import numpy as np
import matplotlib.pyplot as plt

def lagrange(x, X, Y):
    y = []
    for xi in x:
        result = 0
        for i in range(len(X)):
            prod = 1
            for j in range(len(X)):
                if (i!=j and xi!=X[j]):
                    prod = prod * ((xi - X[j])/(X[i]-X[j]))
            prod = prod * Y[i]
            result = result + prod
        y.append(result)
    return np.asarray(y)


uniform_dist = np.random.uniform(low=0.0, high=4*np.pi, size=120)
uniform_dist_y = np.cos(uniform_dist + 3*np.pi/10)
plt.scatter(uniform_dist, uniform_dist_y, c = 'b')
uniform_dist_train = uniform_dist[:100]
uniform_dist_test = uniform_dist[100:]
uniform_dist_y_train = uniform_dist_y[:100]
uniform_dist_y_test = uniform_dist_y[100:]
uniform_dist_pred = lagrange(uniform_dist_test, uniform_dist_train, uniform_dist_y_train)
uniform_dist_train_pred = lagrange(uniform_dist_train, uniform_dist_train, uniform_dist_y_train)
# Train Error
np.mean(np.abs(uniform_dist_train_pred - uniform_dist_y_train))
# Test Error
np.mean(np.abs(uniform_dist_pred - uniform_dist_y_test))




mu = 2*np.pi - 3*np.pi/10
sigma = np.pi/6
normal_dist_pb6 = np.random.normal(mu, sigma, 120)
normal_dist_pb6 = normal_dist_pb6[normal_dist_pb6 >= 0]
normal_dist_pb6 = normal_dist_pb6[normal_dist_pb6 <= 4 * np.pi]
normal_dist_pb6_y = np.cos(normal_dist_pb6 + 3*np.pi/10)
plt.scatter(normal_dist_pb6, normal_dist_pb6_y, c = 'b')
normal_dist_pb6_train = normal_dist_pb6[0:100]
normal_dist_pb6_test = normal_dist_pb6[100:]
normal_dist_pb6_y_train = normal_dist_pb6_y[0:100]
normal_dist_pb6_y_test = normal_dist_pb6_y[100:]
normal_dist_pb6_pred = lagrange(normal_dist_pb6_test, normal_dist_pb6_train, normal_dist_pb6_y_train)
normal_dist_pb6_train_pred = lagrange(normal_dist_pb6_train, normal_dist_pb6_train, normal_dist_pb6_y_train)
# Train Error
np.mean(np.abs(normal_dist_pb6_train_pred - normal_dist_pb6_y_train))
# Test Error
np.mean(np.abs(normal_dist_pb6_pred - normal_dist_pb6_y_test))




mu = 2*np.pi - 3*np.pi/10
sigma = np.pi/4
normal_dist_pb4 = np.random.normal(mu, sigma, 120)
normal_dist_pb4 = normal_dist_pb4[normal_dist_pb4 >= 0]
normal_dist_pb4 = normal_dist_pb4[normal_dist_pb4 <= 4 * np.pi]
normal_dist_pb4_y = np.cos(normal_dist_pb4 + 3*np.pi/10)
plt.scatter(normal_dist_pb4, normal_dist_pb4_y, c = 'b')
normal_dist_pb4_train = normal_dist_pb4[:100]
normal_dist_pb4_test = normal_dist_pb4[100:]
normal_dist_pb4_y_train = normal_dist_pb4_y[:100]
normal_dist_pb4_y_test = normal_dist_pb4_y[100:]
normal_dist_pb4_pred = lagrange(normal_dist_pb4_test, normal_dist_pb4_train, normal_dist_pb4_y_train)
normal_dist_pb4_train_pred = lagrange(normal_dist_pb4_train, normal_dist_pb4_train, normal_dist_pb4_y_train)
# Train Error
np.mean(np.abs(normal_dist_pb4_train_pred - normal_dist_pb4_y_train))
# Test Error
np.mean(np.abs(normal_dist_pb4_pred - normal_dist_pb4_y_test))




mu = 2*np.pi - 3*np.pi/10
sigma = np.pi/2
normal_dist_pb2 = np.random.normal(mu, sigma, 120)
normal_dist_pb2 = normal_dist_pb2[normal_dist_pb2 >= 0]
normal_dist_pb2 = normal_dist_pb2[normal_dist_pb2 <= 4 * np.pi]
normal_dist_pb2_y = np.cos(normal_dist_pb2 + 3*np.pi/10)
plt.scatter(normal_dist_pb2, normal_dist_pb2_y, c = 'b')
normal_dist_pb2_train = normal_dist_pb2[:100]
normal_dist_pb2_test = normal_dist_pb2[100:]
normal_dist_pb2_y_train = normal_dist_pb2_y[:100]
normal_dist_pb2_y_test = normal_dist_pb2_y[100:]
normal_dist_pb2_pred = lagrange(normal_dist_pb2_test, normal_dist_pb2_train, normal_dist_pb2_y_train)
normal_dist_pb2_train_pred = lagrange(normal_dist_pb2_train, normal_dist_pb2_train, normal_dist_pb2_y_train)
# Train Error
np.mean(np.abs(normal_dist_pb2_train_pred - normal_dist_pb2_y_train))
# Test Error
np.mean(np.abs(normal_dist_pb2_pred - normal_dist_pb2_y_test))

