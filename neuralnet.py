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
        self.iter_thresh = 8
        self.act_thresh  = 0.5
        
        self.class_label = np.unique(np.array(self.y))
        self.classes     = np.arange(len(self.class_label))
        self.n_class     = len(self.classes)
        
        self.nn_architecture = nn_architecture
        self.alpha = 1 * np.ones(len(self.nn_architecture))        
        self.lmbda = 10 * np.ones(len(self.nn_architecture))
        
        if self.nn_architecture[-1] != self.n_class:
            print 'nn_architecture[-1]:', self.nn_architecture[-1],\
                  '!=', self.n_class, 'n_classes. Apppending n_classes',\
                  'to nn_architecture.'
            self.nn_architecture = np.append(self.nn_architecture, self.n_class)
          
        self.Y = self.vectorize(self.y, self.n, self.n_class)
        
#        self.alpha = np.arange(1, 0.2 / len(self.nn_architecture), -0.2)
        self.theta_size = self.nn_architecture[0] * (self.m+1)
    
        for i in range(self.nn_architecture.shape[0]):
            if i != 0:
                self.theta_size += self.nn_architecture[i] * (self.nn_architecture[i-1]+1)
    
        self.theta = 0.1 * np.random.rand(self.theta_size)
            
    def vectorize(self, y, n_rows, n_cols):
        y = y.ravel()
        unique_y = np.unique(y)        
        Y = np.zeros([n_rows, n_cols])
        for i in range(n_cols):
            Y[np.where(y == unique_y[i]), i] = 1
        return Y
                
    def appendOnes(self, arr):
#        print 'Appending columns of ones (bias terms)
        temp = np.ones([arr.shape[0], arr.shape[1]+1])
        temp[:, 1:] = arr
        return temp
        
    def h_theta(self, data, theta, append_ones=True):
        '''Data and theta are numpy arrays'''
        if(append_ones):
            data=self.appendOnes(data)
        h = data.dot(theta.T)
        return h
    
    def sigmoid(self, data):
        return (np.add(1, np.exp(-data)))**(-1)

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
        output  = self.sigmoid(output)
        pred_cl = output.argmax(axis=1)
        
        print pred_cl.shape
        return pred_cl#self.class_label[self.classes[pred_cl]]
        
        
    def activation(self, data):
        data = self.sigmoid(data)
#        data[data>self.act_thresh] = 1
#        data[data<=self.act_thresh] = 0
        return data
    
    def calc_error(self, output, Y, is_vector=False):
        if is_vector==False:
                print 'vectorizing Y', Y.shape
                Y = self.vectorize(Y, Y.shape[0], len(np.unique(Y)))
                print 'vectorizing output', output.shape
                output = self.vectorize(output, output.shape[0], len(np.unique(output)))
        print 'err:', np.sum(0.5 * (Y-output) ** 2)/output.shape[0]
        return np.sum(0.5 * (Y-output) ** 2)/output.shape[0]
            
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
            layer_data = self.h_theta(layer_data, layer_theta, append_ones=True)
            nn_data = np.append(nn_data, layer_data.ravel())
            layer_data = self.activation(layer_data)
            if(i!=num_layers-1):
                layer_theta = theta[lower:lower+nn_architecture[i+1]*(nn_architecture[i]+1)]
                layer_theta = layer_theta.reshape(nn_architecture[i+1], nn_architecture[i]+1)                
                lower = nn_architecture[i+1]*(nn_architecture[i]+1)
        return nn_data
     
    def back_propagation(self, nn_architecture, alpha, lmbda, data, theta, Y, append_ones=True):
        num_layers = nn_architecture.shape[0]
        
        nn_data = self.fwd_propagation(nn_architecture, data, theta, append_ones)
        upper_t = theta.shape[0]
        upper_d = nn_data.shape[0]
        n_0     = data.shape[0]
        delta   = np.zeros([n_0, nn_architecture[-1]])
        lower_d = upper_d - (delta.shape[0] * delta.shape[1])
        
        output  = nn_data[lower_d:upper_d]
        output  = output.reshape(delta.shape)
        output  = self.activation(output)
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
        delta   = (Y-output)
        upper_d = lower_d
#        print out2
#        print output
        cost    = -np.sum((Y-output) ** 2)/n_0

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
            layer_data  = self.activation(layer_data)
            
            cost        = cost + lmbda[-(i+1)] * np.sum(layer_theta ** 2)
            
            input_data  = np.ones([n_0, n_nodes_b+1])
            input_data[:, 1:] = layer_data
            
            big_delta   = delta.T.dot(input_data)/n_0
            big_delta[:, 1:] =  lmbda[-(i+1)] * layer_theta[:, 1:]/n_0
            delta       = delta.dot(layer_theta[:, 1:]) * self.der_sigmoid(layer_data)
            
            new_theta   = layer_theta - alpha[i] * big_delta

            theta[lower_t:upper_t] = new_theta.ravel()            
            
            output      = layer_data
           
            upper_t     = lower_t
            upper_d     = lower_d
        print cost
        return cost, theta
    
    def train(self, data, y, append_ones=False, c_valid=False):
        if(c_valid == False):
            print 'Actual learning (not cross validation.',\
                  'Using self.x and self.y'
            data = self.x
            y    = self.y
            append_ones = False
        Y =  self.vectorize(y, data.shape[0], len(np.unique(y)))
        
        i = 0
        cost = 1000
        cost_vec = []
        theta = self.theta
        
        while np.abs(cost) > self.err_thresh and i<self.iter_thresh:
            cost, theta = self.back_propagation(archi, self.alpha, self.lmbda, data, theta, Y, append_ones) 
            i += 1
            cost_vec = np.append(cost_vec, cost)
        
        
        self.theta = theta
        print 'iter:', i, 'final cost:', cost
        if c_valid==False:
            plt.plot(cost_vec, label='full train')
            plt.axis([0, self.iter_thresh, 0.9 * min(cost_vec), 1.1 * max(cost_vec)])
            plt.legend(loc='lower right', shadow=True)
            plt.show()
        else:
            return cost_vec
    
    def accuracy(self, predict, y):
        '''Caclulate accuracy of classification.
        
        y      : (nx1 array) Actual values
        predict: (nx1 array) Predicted values    
        
        Note: n is then number of examples.'''
        
        assert len(y) == len(predict)
        assert len(y) != 0 and len(predict) != 0
        corr = 0
        for i in range(len(y)):
            if int(y[i]) == int(predict[i]):
                corr += 1
        acc = float(corr) / len(y)
        return acc
    
if __name__ == '__main__':
    x=np.load('x.npy')[:1000, :] 
    y=np.load('y.npy')[:1000]
    archi = np.array([25, 10])
    
    archi[archi.shape[0]-1] = len(np.unique(y))
    
    nn = NeuralNetwork(archi, x, y, 0.2)
#    nn.do_kfold_cross_validation()
    nn.train(x, y, append_ones=True)
#    pred = nn.predict(x)
    nn_data = nn.fwd_propagation(nn.nn_architecture, nn.x, nn.theta)
    out = nn_data[-(nn.x.shape[0] * nn.nn_architecture[-1]):]
    out = out.reshape(nn.x.shape[0], nn.nn_architecture[-1])
    out = nn.activation(out)
    pred = out.argmax(axis=1)
    print nn.accuracy(pred, nn.y)