import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_data(ndev=1000):
    '''
    Loads data and splits into `42000 - ndev` inner train set and `ndev` inner test set
    '''
    d = pd.read_csv('digit-recognizer/train.csv')
    d = d.to_numpy()
    nsamples, npixels = d.shape
    np.random.shuffle(d) # in-place shuffle

    d_dev = d[:ndev]
    y_dev = d_dev[:, 0] # (1000,)
    X_dev = d_dev[:, 1:].T/255 # (784, 1000)
    d_train = d[ndev:]
    y_train = d_train[:, 0] # (41000,)
    X_train = d_train[:, 1:].T/255 # (784, 41000)

    return y_dev, X_dev, y_train, X_train

def plot(X):
    img = X_train[3, :]
    img = img.reshape((28, 28))
    plt.gray()
    plt.imshow(img, interpolation='nearest')
    plt.show()

def init_params():
    '''
    Initializes the weights and biases of the first and second layers
    '''
    W1 = np.random.random(size=(16, 784))
    b1 = np.random.random(size=(16, 1))
    W2 = np.random.random(size=(10, 16))
    b2 = np.random.random(size=(10, 1))
    return W1, b1, W2, b2

def ReLU(z):
    '''
    Returns element-wise maximum between 0 and a vector `z`
    '''
    return np.maximum(0, z)

def ReLU_diff(z):
    '''
    Returns derivative of the ReLU function
    '''
    return z > 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_diff(z):
    return np.exp(-z) / (1 + np.exp(-z))**2

def forward_prop(W1, b1, W2, b2, x):
    z1 = W1 @ x + b1 # (16, nsamples)
    a1 = ReLU(z1)
    z2 = W2 @ a1 + b2 # (10, nsamples)
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def one_hot(y):
    nsamples = y.size
    one_hot_y = np.zeros((nsamples, y.max() + 1))
    one_hot_y[np.arange(nsamples), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

def backward_prop(z1, a1, z2, a2, W1, W2, x, y):
    nsamples = y.size
    da2 = 2*(a2 - one_hot(y)) # (10, nsamples)
    dW2 = 1/nsamples*(sigmoid_diff(z2)*da2 @ a1.T) # (10, 16)
    db2 = 1/nsamples*np.sum(sigmoid_diff(z2)*da2, axis=1, keepdims=True) # (10, 1)
    da1 = W2.T @ (sigmoid_diff(z2)*da2) # (16, nsamples)
    dW1 = 1/nsamples*(ReLU_diff(z1)*da1 @ x.T) # (16, 784)
    db1 = 1/nsamples*np.sum(ReLU_diff(z1)*da1, axis=1, keepdims=True) # (16, 1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha*dW1
    b1 -= alpha*db1
    W2 -= alpha*dW2
    b2 -= alpha*db2
    return W1, b1, W2, b2

def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size

def get_predictions(a2):
    return np.argmax(a2, axis=0)

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print(f'Iteration: {i}')
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

if __name__ == '__main__':
    # np.random.seed(0)
    y_dev, X_dev, y_train, X_train = load_data()
    W1, b1, W2, b2 = gradient_descent(X_train, y_train, 0.10, 500)
