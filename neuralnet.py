# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:48:17 2015

@author: neeth
"""
import numpy as np
import matplotlib.pyplot as plt
from brain import Brain

class NeuralNetwork(Brain):
#    shape of data is n_0xm
#    (n_0 is number of examples, m is number of features)
#    Number of nodes in hidden layer i = n_i
#    shape of theta for layer_i = n_i x n_(i-1) + 1 
#    shape of output at layer_i = n_0 X n_i

    def __init__(self, nn_architecture, data, y, c_ratio):
        super(NeuralNetwork, self).__init__(data, y, c_ratio)
        
        self.err_thresh  = 0.4
        self.iter_thresh = 100
        self.act_thresh  = 0.5
        
        self.class_label = np.unique(np.array(self.y))
        self.classes     = np.arange(len(self.class_label))
        self.n_class     = len(self.classes)
        
        self.nn_architecture = nn_architecture
        
        if self.nn_architecture[-1] != self.n_class:
            print 'nn_architecture[-1]:', self.nn_architecture[-1],\
                  '!=', self.n_class, 'n_classes. Apppending n_classes',\
                  'to nn_architecture.'
            self.nn_architecture = np.append(self.nn_architecture, self.n_class)
          
        self.Y = self.vectorize(self.y, self.n, self.n_class)
        
        self.alpha = np.arange(1, 0.2 / len(self.nn_architecture), -0.2)
        self.theta_size = self.nn_architecture[0] * (self.m+1)
    
        for i in range(self.nn_architecture.shape[0]):
            if i != 0:
                self.theta_size += self.nn_architecture[i] * (self.nn_architecture[i-1]+1)
    
        self.theta = 0.1 * np.random.rand(self.theta_size)
        
    def train(self, data, y, append_ones=False, c_valid=False):
        if(c_valid == False):
            print 'Actual learning (not cross validation.',\
                  'Using self.x and self.y'
            data = self.x
            y    = self.y
            append_ones = False
        Y = self.vectorize(y, data.shape[0], len(np.unique(y)))

        i = 0
        cost = 1000
        cost_vec = []
        theta = np.random.rand(self.theta_size)
        
        while cost > self.err_thresh and i<self.iter_thresh:
            cost, theta = self.back_propagation(self.nn_architecture, self.alpha, data, theta, Y, append_ones)
            i += 1
            cost_vec = np.append(cost_vec, cost)
        
        self.theta = theta
        print 'iter:', i, 'final cost:', cost
        plt.plot(cost_vec)
        plt.axis([0, self.iter_thresh, 0, max(cost_vec)])
        plt.show()
        
        return cost, theta
        
    def vectorize(self, y, n_rows, n_cols):
        y = y.ravel()
        unique_y = np.unique(y)        
        Y = np.zeros([n_rows, n_cols])
        for i in range(n_cols):
            Y[np.where(y == unique_y[i]), i] = 1
        return Y
                
    def appendOnes(self, arr):
#        print 'Appending columns of ones (bias terms)'
        temp = np.ones([arr.shape[0], arr.shape[1]+1])
        temp[:, 1:] = arr
        return temp
        
    def sigmoid(self, data, theta, append_ones=True):
        '''Data and theta are numpy arrays'''
        if(append_ones):
            data=self.appendOnes(data)
        h = data.dot(theta.T)
        return (np.add(1, np.exp(-h)))**(-1)
    
    def der_sigmoid(self, a):
        return a * (1-a)
    
    def predict(self, data, append_ones=True, c_valid=True):
        nn_data = self.fwd_propagation(self.nn_architecture, data, self.theta, append_ones)
        n_0     = data.shape[0]        
        n_outs  = self.nn_architecture[-1]
        upper_d = nn_data.shape[0]
        lower_d = upper_d - (n_0 * self.n_class)
        output  = nn_data[lower_d:upper_d]
        output  = output.reshape(n_0, n_outs)
        pred_cl = output.argmax(axis=1)
        
        return self.class_label[self.classes[pred_cl]]
        
        
    def activation(self, data):
        data[data>self.act_thresh] = 1
        data[data<=self.act_thresh] = 0
        return data
        
    def fwd_propagation(self, nn_architecture, data, theta, append_ones=True):
        '''nn_architecture is num_layers x 1 where i-th element is the number
        of nodes in layer_i
        Do not include input layer, but include output layer
        theta is a 1d array with all weights for all layers
        !!!Data should not have bias term'''
        
        num_layers = nn_architecture.shape[0]
        
        layer_data  = data
        layer_theta = theta[:nn_architecture[0]*(data.shape[1]+1)]
        layer_theta = layer_theta.reshape(nn_architecture[0], data.shape[1]+1)
        lower = (data.shape[1]+1)*nn_architecture[0]
        
        nn_data = np.append([], layer_data.ravel())
        for i in range(num_layers):
            layer_data = self.sigmoid(layer_data, layer_theta, True)
            nn_data = np.append(nn_data, layer_data.ravel())
            if(i!=num_layers-1):
                layer_theta = theta[lower:lower+nn_architecture[i+1]*(nn_architecture[i]+1)]
                layer_theta = layer_theta.reshape(nn_architecture[i+1], nn_architecture[i]+1)                
                lower = nn_architecture[i+1]*(nn_architecture[i]+1)
        
        return nn_data
     
    def calc_error(self, output, Y, is_vector=False):
        if is_vector==False:
                print 'vectorizing Y'
                Y = self.vectorize(Y, Y.shape[0], len(np.unique(Y)))
                print 'vectorizing output'
                output = self.vectorize(output, output.shape[0], len(np.unique(output)))
        print 'err:', np.sum(0.5 * (Y-output) ** 2)/output.shape[0]
        return np.sum(0.5 * (Y-output) ** 2)/output.shape[0]
        
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

###############################################################################
#       So, here I am trying to create an output matrix (n_0 X 10),
#       where, in every row, the column with the highest value in 
#       output is 1 and everything else is 0.
#       What I am doing here is highly in-efficient. So, if either
#       of you know any python/numpy specific way to achieve this,
#       make the changes here.
#       I am new to python! :(
        out2    = np.zeros(output.shape)
        indices = output.argmax(axis=1)
        for i in range(indices.shape[0]):
            out2[i, indices[i]] = 1
###############################################################################
        delta   = (Y-out2) #* self.der_sigmoid(output)
#        print 'delta'
        print delta
        upper_d = lower_d

        cost    = np.sum(0.5 * (Y-out2) ** 2)
        
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
            new_theta   = layer_theta - alpha[i] * big_delta
            
            theta[lower_t:upper_t] = new_theta.ravel()            
            
            output      = layer_data
           
            upper_t     = lower_t
            upper_d     = lower_d
        
        return cost, theta
    
    
if __name__ == '__main__':
#    x=np.load('x.npy') 
#    y=np.load('y.npy')
    x=np.array([[1], [2]])
    y=np.array([[1], [2]])
#    y[-1] = y[-2]
#    print x
#    print y
#    print y
#    print x.shape, y.shape
    archi = np.array([2])
    
    archi[archi.shape[0]-1] = len(np.unique(y))
    
    nn = NeuralNetwork(archi, x, y, 0.2)
#    cost, theta = nn.train(x, y, append_ones=True)
    i = 0#nn.iter_thresh-1
    cost_vec = []
    while i<nn.iter_thresh:
#        nn_data = nn.fwd_propagation(archi, x, nn.theta, append_ones=True)
        cost, theta = nn.back_propagation(archi, nn.alpha, x, nn.theta, nn.Y, append_ones=True)
        print cost
        cost_vec = np.append(cost_vec, cost)
        
        i += 1
#    print cost_vec
    pred = nn.predict(x)
    print pred
#    print np.max(cost_vec)
#    plt.plot(cost_vec)
#    plt.axis([0, i, np.min(cost_vec)-1, np.max(cost_vec)+1])
#    plt.show()

    '''a = np.zeros([3, 2])
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
#    y = np.array([y.shape])
    for i in range(len(y)):
        if i == 2:
            y[i] = 1000
        else:
            y[i] = 10

#    y[-1]=1
#    Y = np.zeros([a.shape[0], archi[-1]])
#    for i in range(y.shape[0]):
#        Y[i, y[i]] = 1
#        
        
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[i][j] = i*a.shape[1]+j
    
    nn = NeuralNetwork(archi, a, y, 0.5)
    cost, cross_err_mat = nn.do_kfold_cross_validation()
    print 'done'
    '''
#    cost, theta = nn.train(a, y)
#    print nn.predict(a, theta)
#    cost = 1000
#    cost_vec = [];
#    i = 0
#    while cost > 0.1 and i <1000:
#        cost, theta = nn.back_propagation(archi, alpha, a, theta, Y);
##        print theta
#        cost_vec.append(cost)
#        i += 1
##        print len(cost_vec)
#    plt.plot(cost_vec)
#    plt.axis([0, 1000, 0, 1])
#    plt.show()'''
        
    