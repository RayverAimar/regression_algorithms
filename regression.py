import numpy as np
import matplotlib.pyplot as plt

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
            if abs(cost - prev_cost) < 0.0001: #1e-6
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

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

clf = LinearRegression()
clf.fit(X,y)
y_pred = clf.predict(X.reshape(np.size(X), -1))
print(y_pred)
