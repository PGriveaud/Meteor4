#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:19:49 2020

@author: pipoune
"""


import numpy as np

class MLP(object):

    def __init__(self, num_inputs, hidden_layers, num_outputs):


        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        layers = [num_inputs] + hidden_layers + [num_outputs]

        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    def forward_propagate(self, inputs):

        self.activations[0] = inputs

        for i, w in enumerate(self.weights):

            hidden = np.dot(self.activations[i], w)
            self.activations[i+1] = self._sigmoid(hidden)

        return self.activations[-1]


    def back_propagate(self, error):

        for i in reversed(range(len(self.derivatives))):

            delta = error * self._sigmoid_derivative(self.activations[i+1])
 
            self.derivatives[i] = np.dot(self.activations[i].reshape(-1, 1), delta.reshape(-1, 1).T)
            error = np.dot(delta, self.weights[i].T)


    def train(self, inputs, targets, epochs, learning_rate):
         for i in range(epochs):
            sum_errors = 0

            for j, input in enumerate(inputs):
                target = targets[j]

                output = self.forward_propagate(input)

                error = output - target

                self.back_propagate(error)

                self.gradient_descent(learning_rate)

                sum_errors += self._mse(target, output)

            print("Error: {} at epoch {}".format(sum_errors / len(inputs), i+1))


    def gradient_descent(self, learningRate=1):

        for i in range(len(self.weights)):
            weights = self.weights[i]
            weights -= self.derivatives[i] * learningRate
            self.weights[i] = weights

    def _sigmoid(self, x):

        return 1.0 / (1 + np.exp(-x))


    def _sigmoid_derivative(self, x):

        return x * (1.0 - x)


    def _mse(self, target, output):

        return np.average((target - output) ** 2)

if __name__ == "__main__":


    N = 1000
    L = 2
    x = np.random.uniform(low=0.0, high=0.5, size=(N,2))
    y = np.sum(x,axis=1).reshape(-1,1)

    mlp = MLP(2, [3], 1)
    mlp.train(x, y, 100, 0.1)

    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    output = mlp.forward_propagate(input)

    
    print("{} + {} = {}".format(input[0], input[1], output[0]))


