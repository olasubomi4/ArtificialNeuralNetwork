import numpy as np


class ArtificialNeuralNetwork:

    def __init__(self, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
        self.weights_input_hidden = weights_input_hidden
        self.weights_hidden_output = weights_hidden_output
        self.bias_hidden = bias_hidden
        self.bias_output = bias_output

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def loss_function(self, target, prediction):
        return np.mean(np.square(target - prediction))

    def calculate_error(self, target, prediction):
        return prediction - target

    def forward_pass(self, inputs):
        # Hidden layer
        hidden_layer_linear_output = self.linear_combination(self.weights_input_hidden, inputs, self.bias_hidden)
        hidden_layer_activation_output = self.sigmoid(hidden_layer_linear_output)

        # Output layer
        output_layer_linear_output = self.linear_combination(self.weights_hidden_output, hidden_layer_activation_output,
                                                             self.bias_output)
        prediction = self.sigmoid(output_layer_linear_output)
        return prediction

    def backward_pass(self, target, prediction, inputs, learning_rate):
        output_layer_error = self.calculate_error(target, prediction)
        output_layer_delta = output_layer_error * self.sigmoid_derivative(prediction)

        hidden_layer_error = output_layer_delta * self.weights_hidden_output
        hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(self.hidden_layer_activation_output)

        # Gradient updates
        self.weights_input_hidden -= learning_rate * np.outer(hidden_layer_delta, inputs)
        self.weights_hidden_output -= learning_rate * output_layer_delta * self.hidden_layer_activation_output

    def train(self, inputs, target, learning_rate, epochs):
        for epoch in range(epochs):
            prediction = self.forward_pass(inputs)
            self.backward_pass(target, prediction, inputs, learning_rate)

            if epoch % 10 == 0:
                loss = self.loss_function(target, prediction)
                print(f"Epoch {epoch}, Loss: {loss}")

    def linear_combination(self, weights, inputs, bias):
        return np.dot(inputs, weights.T) + bias

    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))


# Example usage
target_output = np.array([1])
weights_input_hidden = np.array([[0.1, 0.2], [0.3, 0.4]])
weights_hidden_output = np.array([0.5, 0.6])
input_data = np.array([2, 3])
bias_hidden = np.array([0.1, 0.2])
bias_output = np.array([0.3])

ann = ArtificialNeuralNetwork(weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
ann.train(input_data, target_output, learning_rate=0.1, epochs=100)
