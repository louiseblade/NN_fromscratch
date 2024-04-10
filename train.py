from Layer import Layer_Dense
from Activation_function import Activation_ReLU, Activation_Softmax
from data_generate import spiral_data, vertical_data
from Loss_function import Loss_CategoricalCrossentropy
from optimizer import CustomOptimizer
import numpy as np


if __name__ == '__main__':
    # Create dataset
    X, y = spiral_data(points=100, classes=3)

    # Create Dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2, 64)
    activation1 = Activation_ReLU()

    # Create second Dense layer with 3 input features (as we take output of previous layer here) and 3 output values
    dense2 = Layer_Dense(64, 3)
    activation2 = Activation_Softmax()

    # Create loss function
    loss_function = Loss_CategoricalCrossentropy()

    # Create optimizer
    optimizer = CustomOptimizer(learning_rate=0.05)

    # Train in loop
    for epoch in range(10001):
        # Perform a forward pass of our training data through this layer
        dense1.forward(X)
        activation1.forward(dense1.output)

        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        # Calculate loss from output of activation2 so softmax activation
        data_loss = loss_function.calculate(activation2.output, y)

        # Calculate accuracy from output of activation2 and targets
        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions == y)

        if epoch % 100 == 0:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {data_loss:.3f}')

        # Backward pass
        loss_function.backward(activation2.output, y)
        activation2.backward(loss_function.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights and biases
        optimizer.update_parameters([dense1, dense2])
