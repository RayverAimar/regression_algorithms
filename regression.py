import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def gradient_descent_visualization():
    def y_function(x):
        return x ** 2

    def y_derivative(x):
        return 2 * x

    X = np.arange(-100, 100, 1)
    y = y_function(X)

    current_pos = (80, y_function(80))

    learning_rate = 0.01

    prev_pos = float('inf')
    for i in range(1000):
        new_x = current_pos[0] - learning_rate * y_derivative(current_pos[0])
        new_y = y_function(new_x)
        current_pos = (new_x, new_y)
        if abs(current_pos[0] - prev_pos) < 1e-3:
            print(f"Converged at iteration {i}")
            break
        prev_pos = current_pos[0]
        plt.plot(X,y)
        plt.scatter(current_pos[0], current_pos[1], c='red')
        plt.pause(0.001)
        plt.clf()

class LinearRegression():
    
    def __init__(self, learning_rate=0.001, iters=1000) -> None:
        self.learning_rate = learning_rate
        self.iters = iters
        self.theta = None
    
    def fit(self, X,y, subplots=False):
        if not isinstance(X, np.ndarray):
            raise TypeError('"X" must be an ndarray to perform regression')
        if not isinstance(y, np.ndarray):
            raise TypeError('"y" must be an ndarray to perform regression')
        if len(X.shape) == 1 or X.shape[1] > 1:
            # X needs to be one dimensional horizontally
            X = X.reshape((np.size(X), -1))
        if len(y.shape) == 1 or y.shape[1] > 1:
            # y needs to be one dimensional horizontally
            y = y.reshape((np.size(y), -1))
        if y.size != X.size:
            raise TypeError('"X" and "y" must be of equal size')
        m = X.size
        X = np.hstack((np.ones(shape=(m, 1)), X))
        self.theta = np.zeros((2,1))
        prev_cost = float('inf')
        for i in range(self.iters):
            y_pred = np.dot(X,self.theta)
            cost = (1/(2*m))*np.sum(np.square(y_pred - y))
            theta_derivative = (1/m)*np.dot(X.T, y_pred - y)
            self.theta = self.theta - self.learning_rate * theta_derivative
            if abs(cost - prev_cost) < 0.0001: #1e-4
                print(f'Converged in iteration {i}')
                break
            prev_cost = cost
            if subplots:
                plt.scatter(X[:,1],y, c='red')
                plt.plot(X[:,1],y_pred)
                plt.pause(0.001)
                plt.clf()
            
    def predict(self, X):
        X = np.hstack((np.ones(shape=(X.size, 1)), X))
        y_pred = np.dot(X,self.theta)
        return y_pred

class PolynomialRegression():

    def __init__(self, learning_rate=0.000000005, iters=10000, degree=None) -> None:
        self.learning_rate = learning_rate
        self.iters = iters
        self.theta = None
        self.degree = degree + 1

    def fit(self, X,y):
        if not isinstance(X, np.ndarray):
            raise TypeError('"X" must be an ndarray to perform regression')
        if not isinstance(y, np.ndarray):
            raise TypeError('"y" must be an ndarray to perform regression')
        if len(X.shape) == 1:
            raise TypeError("X shall not be one-dimensional to perform polynomial regression")
        if self.degree == None:
            self.degree = X.shape[1]
        else:
            X = np.hstack((X, np.ones((X.shape[0], self.degree - 1 - X.shape[1]))))
        if y.shape[0] != X.shape[0]:
            raise TypeError('"X" and "y" must be of equal size')
        m, n = X.shape
        X = np.hstack((np.ones(shape=(m, 1)), X))
        for i in range(2, self.degree):
            X[:,i] = X[:,1]**i
        self.theta = np.zeros((n+1,1))
        prev_cost = float('inf')
        costs = []
        for i in range(self.iters):
            y_pred = np.dot(X,self.theta)
            '''
            if i%30 == 0:
                print(i)
                print("y_pred")
                print(y_pred)
                print("theta")
                print(self.theta)
            '''
            cost = (1/(2*m))*np.sum(np.square(y_pred - y))
            theta_derivative = (1/m)*np.dot(X.T, y_pred - y)
            self.theta = self.theta - self.learning_rate * theta_derivative
            costs.append(cost)
            if i%100 == 0:
                print("Cost:", cost)
            if abs(cost - prev_cost) < 0.0000001: #1e-4
                print(f'Converged in iteration {i}')
                break
            prev_cost = cost
        return costs
            
    def predict(self, X):
        X = np.hstack((np.ones(shape=(X.shape[0], 1)), X))
        X = np.hstack((X, np.ones((X.shape[0], self.degree - X.shape[1]))))
        for i in range(2, self.degree):
            X[:,i] = X[:,1]**i
        print(X)
        print(y)
        y_pred = np.dot(X,self.theta)
        return y_pred
    

class MultilinearRegression():
    
    def __init__(self, learning_rate=0.00000000001, iters=10000) -> None:
        self.learning_rate = learning_rate
        self.iters = iters
        self.theta = None

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            raise TypeError('"X" must be an ndarray to perform regression')
        if not isinstance(y, np.ndarray):
            raise TypeError('"y" must be an ndarray to perform regression')
        if len(X.shape) == 1:
            raise TypeError("X shall not be one-dimensional to perform Multilinear regression")
        m, n = X.shape
        X = np.hstack((np.ones(shape=(X.shape[0], 1)), X))
        self.theta = np.zeros((n+1,1))
        for i in range(self.iters):
            y_pred = np.dot(X,self.theta)
            cost = (1/(2*m))*np.sum(np.square(y_pred - y))
            theta_derivative = (1/m)*np.dot(X.T, y_pred - y)
            self.theta = self.theta - self.learning_rate * theta_derivative
            if (i%(self.iters/10) == 0):
                print("Cost is:", cost)


'''
df = pd.read_csv('../train_data.csv')
train = df.drop(['Unnamed: 0','Id'], axis=1)
train = train.values
X = train[:,:-1]
y = train[:,-1].reshape(train.shape[0], 1)

clf = MultilinearRegression()
clf.fit(X,y)
'''

#gradient_descent_visualization()
'''
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

clf = LinearRegression()
clf.fit(X,y)
y_pred = clf.predict(X.reshape(np.size(X), -1))
print(y_pred)
'''

# Polynomial Regression
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000])
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

clf = PolynomialRegression(degree=4)
costs = clf.fit(X,y)
print(clf.predict(X.reshape(np.size(X), 1)))

plt.plot(np.arange(0,10000), costs)
plt.show()

plt.scatter(X,y, c="red")
plt.plot(X, clf.predict(X.reshape(np.size(X), 1)))
plt.show()