#"TASK:1

import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.randn(input_size + 1)
        self.learning_rate = learning_rate
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.sigmoid(summation)
    
    def train(self, inputs, target):
        prediction = self.predict(inputs)
        error = target - prediction
        self.weights[1:] += self.learning_rate * error * inputs
        self.weights[0] += self.learning_rate * error


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1]) 

perceptron = Perceptron(input_size=2)
for _ in range(1000):
    for inputs, target in zip(X, y):
        perceptron.train(inputs, target)

test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for inputs in test_inputs:
    print(inputs, perceptron.predict(inputs))



#Task 2:

class MultiFeaturePerceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.randn(input_size + 1)
        self.learning_rate = learning_rate
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.sigmoid(summation)
    
    def train(self, inputs, target):
        prediction = self.predict(inputs)
        error = target - prediction
        self.weights[1:] += self.learning_rate * error * inputs
        self.weights[0] += self.learning_rate * error

X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([0, 1, 1, 1]) 

perceptron = MultiFeaturePerceptron(input_size=3)
for _ in range(1000):
    for inputs, target in zip(X, y):
        perceptron.train(inputs, target)

test_inputs = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
predictions = [perceptron.predict(inputs) for inputs in test_inputs]
print("Predictions:", predictions)

import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
    if y[i] == 0:
        ax.scatter(X[i][0], X[i][1], X[i][2], color='blue')
    else:
        ax.scatter(X[i][0], X[i][1], X[i][2], color='red')


x_vals = np.linspace(0, 1, 10)
y_vals = np.linspace(0, 1, 10)
x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
z_mesh = (-perceptron.weights[0] - perceptron.weights[1] * x_mesh - perceptron.weights[2] * y_mesh) / perceptron.weights[3]
ax.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.5)


ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')

plt.show()



#Task 3

class PerceptronWithBias:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(summation)
    
    def train(self, inputs, target):
        prediction = self.predict(inputs)
        error = target - prediction
        self.weights += self.learning_rate * error * inputs
        self.bias += self.learning_rate * error


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1]) 

perceptron = PerceptronWithBias(input_size=2)
for _ in range(1000):
    for inputs, target in zip(X, y):
        perceptron.train(inputs, target)

test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = [perceptron.predict(inputs) for inputs in test_inputs]
print("Predictions:", predictions)



#Task 4


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X_binary = X[(y == 0) | (y == 1)]
y_binary = y[(y == 0) | (y == 1)]

X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(summation)
    
    def train(self, inputs, target):
        prediction = self.predict(inputs)
        error = target - prediction
        self.weights += self.learning_rate * error * inputs
        self.bias += self.learning_rate * error


perceptron = Perceptron(input_size=X_train.shape[1])
for _ in range(1000):
    for inputs, target in zip(X_train, y_train):
        perceptron.train(inputs, target)


plt.figure(figsize=(8, 6))

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, label="Data Points")

x_vals = np.linspace(4, 7.5, 10)
y_vals = -(perceptron.weights[0] * x_vals + perceptron.bias) / perceptron.weights[1]
plt.plot(x_vals, y_vals, color='black', linestyle='-', label='Decision Boundary')

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Perceptron Decision Boundary on Iris Dataset')
plt.legend()
plt.show()




#Task 5


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron as SklearnPerceptron
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target


X_binary = X[(y == 0) | (y == 1)]
y_binary = y[(y == 0) | (y == 1)]


X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return np.where(self.sigmoid(summation) >= 0.5, 1, 0)
    
    def train(self, inputs, target):
        prediction = self.predict(inputs)
        error = target - prediction
        self.weights += self.learning_rate * error * inputs
        self.bias += self.learning_rate * error

custom_perceptron = Perceptron(input_size=X_train.shape[1])
for _ in range(1000):
    for inputs, target in zip(X_train, y_train):
        custom_perceptron.train(inputs, target)

sklearn_perceptron = SklearnPerceptron()
sklearn_perceptron.fit(X_train, y_train)

custom_accuracy = np.mean(custom_perceptron.predict(X_test) == y_test)
sklearn_accuracy = sklearn_perceptron.score(X_test, y_test)
print("Custom Perceptron Accuracy:", custom_accuracy)
print("Sklearn Perceptron Accuracy:", sklearn_accuracy)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, label="Data Points")
plt.title('Custom Perceptron Decision Boundary')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

x_vals = np.linspace(4, 7.5, 10)
y_vals = -(custom_perceptron.weights[0] * x_vals + custom_perceptron.bias) / custom_perceptron.weights[1]
plt.plot(x_vals, y_vals, color='black', linestyle='-', label='Decision Boundary (Custom)')

plt.subplot(1, 2, 2)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, label="Data Points")
plt.title('Sklearn Perceptron Decision Boundary')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

coef = sklearn_perceptron.coef_[0]
intercept = sklearn_perceptron.intercept_
x_vals_sklearn = np.linspace(4, 7.5, 10)
y_vals_sklearn = -(coef[0] * x_vals_sklearn + intercept) / coef[1]
plt.plot(x_vals_sklearn, y_vals_sklearn, color='black', linestyle='-', label='Decision Boundary (Sklearn)')

plt.legend()
plt.show()

