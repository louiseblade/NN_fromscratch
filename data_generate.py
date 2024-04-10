import numpy as np
import matplotlib.pyplot as plt

def spiral_data(points, classes):
    np.random.seed(0)
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number

    return X, y

def vertical_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype=int)

    for class_number in range(classes):
        print(class_number)
        mean1, mean2 = np.random.uniform(-0.2 + 0.5 * class_number, 0.2 + 0.5 * class_number), np.random.uniform(0.1, 0.5)
        std1, std2 = 0.15, 0.15

        interval1 = np.random.normal(mean1, std1, points)
        interval2 = np.random.normal(mean2, std2, points)

        class_indices = range(points * class_number, points * (class_number + 1))
        X[class_indices] = np.column_stack((interval1, interval2))
        y[class_indices] = class_number

    return X, y

if __name__ == '__main__':

    # X, y = spiral_data(100, 3)
    X, y = vertical_data(100, 3)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
    plt.show()