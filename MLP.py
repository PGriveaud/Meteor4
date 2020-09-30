#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:56:28 2020

@author: pipoune
"""

import numpy as np

class MLP:
  def __init__(self, dim_input, dim_hid, dim_output, num_layers):
    self.dim_input = dim_input
    self.dim_hid = dim_hid
    self.dim_output = dim_output
    self.num_layers = num_layers

    self.activations = []
    self.hidden = []
    self.weight = []
    self.dimensions = [dim_input] + [dim_hid]*(num_layers-1) + [dim_output]   

    self.gradient = []

    for i in range(self.num_layers):
      w = np.random.rand(self.dimensions[i],self.dimensions[i+1])
      #print(f"weight {i} = {w}")
      self.weight.append(w)

  def sigmoid(self, x):
    return 1/(1+np.exp(-x))

  def sigmoid_der(self, x):
    return self.sigmoid(x)*(1-self.sigmoid(x))

  def feedforward(self, input):
    self.bias = []
    self.activations.append(input)
    self.hidden.append(0)  #The term h(1) is set as zero because it doesn't exist

    for i in range(self.num_layers):
      b = np.random.rand(self.dimensions[i+1])
      self.bias.append(b)
      h = np.dot(self.activations[i], self.weight[i]) + self.bias[i]
      a = self.sigmoid(h)
      self.hidden.append(h)
      self.activations.append(a)
      
      output = self.activations[-1] 

    return output

  def error_1(self, y, y_hat):
    J = y_hat - y
    return J

  def backpropagation(self, error):
    for i in reversed(np.arange(1,self.num_layers+1)):
      act = self.activations[i-1]
      delt = error*self.sigmoid_der(self.hidden[i])
      
      act = act.reshape(act.shape[0],-1)
      delt = delt.reshape(delt.shape[0],-1)

      grad = np.dot(act, delt)
      self.gradient.append(grad)
      error = np.dot(delt,np.transpose(self.weight[i-1]))
    return error

  
  def gradient_descent(self, lr):
    self.gradient.reverse()
    for i in range(len(self.weight)):
      self.weight[i] = self.weight[i] - lr*self.gradient[i]

    return self.weight
  
    
  
    

if __name__ == "__main__":

  input_size = 2
  hidden_layers_size = 10
  output_size = 1
  layers_number = 2

  mlp = MLP(input_size, hidden_layers_size, output_size, layers_number)

  x = np.array([[0.2, 0.3], [0.3,0.1], [0.1,0.6], [0.2,0.5]])
  y = [0.5, 0.4, 0.7, 0.7]



  n_epochs = 500
  learning_rate = 1e-4

  for epoch in range(n_epochs):
      running_loss = 0
      for i in range(len(x)):
          data = x[i]
          target = y[i]
          y_hat = mlp.feedforward(data)
          loss = mlp.error_1(target,y_hat)
          # print(type(error[0]))
          backprop = mlp.backpropagation(loss[0])
          train = mlp.gradient_descent(learning_rate)
          
          running_loss += loss.item()
          if epoch % 100 == 99:
                print('[Epoch %d] loss: %.3f' %
                              (epoch + 1, running_loss/len(x)))


Test = [0.3,0.1]
test_output = mlp.feedforward(Test)

print(test_output)
























