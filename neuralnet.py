# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:48:17 2015

@author: neeth
"""
import numpy as np
import matplotlib.pyplot as plt


class neuralnetwork:
#    shape of data is n_0xm
#    (n_0 is number of examples, m is number of features)
#    Number of nodes in hidden layer i = n_i
#    shape of theta for layer_i = n_i x n_(i-1) + 1 
#    shape of output at layer_i = n_0 X n_i

    def appendOnes(self, arr):
        print 'Appending columns of ones (bias terms)'
        temp = np.ones([arr.shape[0], arr.shape[1]+1])
        temp[:, 1:] = arr
        return temp
        
    def sigmoid(self, data, theta, append_ones=False):
        '''Data and theta are numpy arrays'''
        if(append_ones):
            data=self.appendOnes(data)
        h = data.dot(theta.T)
        return (np.add(1, np.exp(-h)))**(-1)
    
    def der_sigmoid(self, a):
        return a * (1-a)
    
    def fwd_propagation(self, nn_architecture, data, theta, append_ones=True):
        '''nn_architecture is num_layers x 1 where i-th element is the number
        of nodes in layer_i
        Do not include input layer, but include output layer
        theta is a 1d array with all weights for all layers
        !!!Data should not have bias term'''
        
        num_layers = nn_architecture.shape[0]
        
        layer_data  = data
        layer_theta = theta[:(data.shape[1]+1)*nn_architecture[0]]
        layer_theta = layer_theta.reshape(nn_architecture[0], data.shape[1]+1)
        lower = (data.shape[1]+1)*nn_architecture[0]
        
        nn_data = np.append([], layer_data)
        for i in range(num_layers):
            layer_data = self.sigmoid(layer_data, layer_theta, True)
            nn_data = np.append(nn_data, layer_data)           
            if(i!=num_layers-1):
                layer_theta = theta[lower:lower+nn_architecture[i+1]*(nn_architecture[i]+1)]
                layer_theta = layer_theta.reshape(nn_architecture[i+1], nn_architecture[i]+1)                
                lower = nn_architecture[i+1]*(nn_architecture[i]+1)
        
        return nn_data
        
    def back_propagation(self, nn_architecture, alpha, data, theta, Y, append_ones=True):
        num_layers = nn_architecture.shape[0]
        
        nn_data = self.fwd_propagation(nn_architecture, data, theta, append_ones)
        upper_t = theta.shape[0]
        upper_d = nn_data.shape[0]
        n_0     = data.shape[0]
        delta   = np.zeros([n_0, nn_architecture[-1]])
        lower_d = upper_d - (delta.shape[0] * delta.shape[1])
        
        output  = nn_data[lower_d:upper_d]
        output  = output.reshape(delta.shape)
        delta   = (Y-output) * self.der_sigmoid(output)
        upper_d = lower_d
        
        cost    = np.sum(0.5 * (Y-output) ** 2)/n_0
        
        layer_data  = np.zeros([1, 1])
        layer_theta = np.zeros([1, 1])
        
        for i in range(num_layers):
            n_nodes = nn_architecture[-(i+1)]
            
            if i<nn_architecture.shape[0]-1:#negative indices can go till arr_size
                n_nodes_b = nn_architecture[-(i+2)]
            else:
                n_nodes_b = data.shape[1]
            
            lower_t     = upper_t - n_nodes * (n_nodes_b+1)
            lower_d     = upper_d - n_0 * n_nodes_b
            
            layer_theta = theta[lower_t:upper_t]
            layer_data  = nn_data[lower_d:upper_d]
            
            layer_theta = layer_theta.reshape(n_nodes, (n_nodes_b+1))
            layer_data  = layer_data.reshape(n_0, n_nodes_b)
            
            input_data  = np.zeros([n_0, n_nodes_b+1])
            input_data[:, 1:] = layer_data
            
            big_delta   = delta.T.dot(input_data) / n_0
            delta       = delta.dot(layer_theta) * self.der_sigmoid(input_data)
            delta       = delta[:, 1:delta.shape[1]]
            new_theta   = layer_theta + alpha[i] * big_delta
            
            theta[lower_t:upper_t] = new_theta.ravel()            
            
            output      = layer_data
           
            upper_t     = lower_t
            upper_d     = lower_d
        
        return cost, theta
        
if __name__ == '__main__':
    a = np.zeros([3, 2])
    archi = np.zeros(3)
    archi.fill(4)
    archi[archi.shape[0]-1] = 2
    alpha = 0.1 * np.ones(archi.shape)
    theta_size = archi[0] * (a.shape[1]+1)
    
    for i in range(archi.shape[0]):
        if i != 0:
            theta_size += archi[i] * (archi[i-1]+1)
    
    theta = np.arange(theta_size)/theta_size
    
    y = np.arange(a.shape[0]).reshape(a.shape[0], 1)
    y[-1]=1
    Y = np.zeros([a.shape[0], archi[-1]])
    for i in range(y.shape[0]):
        Y[i, y[i]] = 1
        
        
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[i][j] = i*a.shape[1]+j
    nn = neuralnetwork()
    cost = 1000
    cost_vec = [];
    i = 0
    while cost > 0.1 and i <1000:
        cost, theta = nn.back_propagation(archi, alpha, a, theta, Y);
#        print theta
        cost_vec.append(cost)
        i += 1
#        print len(cost_vec)
    plt.plot(cost_vec)
    plt.axis([0, 1000, 0, 1])
    plt.show()
        
    