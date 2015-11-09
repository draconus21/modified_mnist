# -*- coding: utf-8 -*-
"""
Created on Sun Nov 08 16:11:40 2015

@author: neeth
"""

import numpy as np
from brain import Brain
import matplotlib.pyplot as plt

class NeuralNetwork2(Brain):
    
    def __init__(self, nn_architecture, data, y, c_ratio):
        super(NeuralNetwork2, self).__init__(data, y, c_ratio)
        
        self.err_thresh  = 0.4
        self.iter_thresh = 25
        self.act_thresh  = 0.5
        
        self.class_label = np.unique(np.array(self.y))
        self.classes     = np.arange(len(self.class_label))
        self.n_class     = len(self.classes)
        
        self.nn_architecture = nn_architecture
        self.lmbda = 100 * np.ones(len(archi))
        
        if self.nn_architecture[-1] != self.n_class:
            print 'nn_architecture[-1]:', self.nn_architecture[-1],\
                  '!=', self.n_class, 'n_classes. Apppending n_classes',\
                  'to nn_architecture.'
            self.nn_architecture = np.append(self.nn_architecture, self.n_class)
          
        self.Y = self.vectorize(self.y, self.n, self.n_class)
        
        self.alpha = np.arange(1, 0.2 / len(self.nn_architecture), -0.2)
#        self.alpha = 1 * np.ones(len(self.nn_architecture))        
        self.theta_size = self.nn_architecture[0] * (self.m+1)
    
        for i in range(self.nn_architecture.shape[0]):
            if i != 0:
                self.theta_size += self.nn_architecture[i] * (self.nn_architecture[i-1]+1)
    
        self.theta = 1 * np.random.rand(self.theta_size)
        
    def h_theta(self, data, theta, append_ones=True):
        '''Data and theta are numpy arrays'''
        if(append_ones):
            data=self.appendOnes(data)
        h = data.dot(theta.T)
        return h
    
    def appendOnes(self, arr):
#        print 'Appending columns of ones (bias terms)
        temp = np.ones([arr.shape[0], arr.shape[1]+1])
        temp[:, 1:] = arr
        return temp
    
    def sigmoid(self, data):
        return (np.add(1, np.exp(-data)))**(-1)
    
    def der_sigmoid(self, a):
       return a * (1-a)   

    def vectorize(self, y, n_rows, n_cols):
        y = y.ravel()
        unique_y = np.unique(y)        
        Y = np.zeros([n_rows, n_cols])
        for i in range(n_cols):
            Y[np.where(y == unique_y[i]), i] = 1
        return Y
    
    def back_propagation(self, nn_archi, alpha, lmbda, data, theta, Y, append_ones=True):
        n_0 = data.shape[0]
        m   = data.shape[1]        
        
        theta1 = theta[:((m+1) * nn_archi[0])]
        theta1 = theta1.reshape(nn_archi[0], m+1)
        
        theta2 = theta[nn_archi[0] * (m+1):]
        theta2 = theta2.reshape(nn_archi[1], nn_archi[0]+1)
        
        theta1_grad = np.zeros(theta1.shape)
        theta2_grad = np.zeros(theta2.shape)
        
        a1 = self.appendOnes(data)
        z2 = self.h_theta(a1, theta1, append_ones=False)
        a2 = self.appendOnes(self.sigmoid(z2))
        z3 = self.h_theta(a2, theta2, append_ones=False)
        a3 = self.sigmoid(z3)
        
        J = -np.sum((Y-a3) ** 2)/n_0
        reg = lmbda[0] * np.sum(np.sum(theta1 ** 2)) + lmbda[1] * np.sum(np.sum(theta2 ** 2))
        J += reg
        
        delta3 = np.zeros([nn_archi[1], n_0])
        delta2 = np.zeros([nn_archi[0], n_0])

        for i in range(n_0):
            delta3[:, i] = a3[i, :] - Y[i, :]
            delta2[:, i] = (theta2[:, 1:].T.dot(delta3[:, i])) * self.der_sigmoid(self.sigmoid(z2[i, :]))
        
        big_delta1 = delta2.dot(a1)
        big_delta2 = delta3.dot(a2)
        
        big_delta1 = big_delta1/n_0
        big_delta2 = big_delta2/n_0
        
        reg1 = lmbda[0] * theta1[:, 1:]/n_0
        reg2 = lmbda[1] * theta2[:, 1:]/n_0
        
        big_delta1[:, 1:] = big_delta1[:, 1:] + reg1
        big_delta2[:, 1:] = big_delta2[:, 1:] + reg2
        
        theta1_grad = big_delta1
        theta2_grad = big_delta2
        
        theta1 = theta1 - alpha[0] * theta1_grad
        theta2 = theta2 - alpha[1] * theta2_grad
        
        theta = np.append(theta1.ravel(), theta2.ravel())
        
        return J, theta
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
        
        while cost > self.err_thresh and i<self.iter_thresh:
            cost, theta = self.back_propagation(archi, alpha, lmbda, data, theta, Y, append_ones) 
            i += 1
            cost_vec = np.append(cost_vec, cost)
            
        self.theta = theta
        print 'iter:', i, 'final cost:', cost
        if c_valid==False:
            plt.plot(cost_vec, label='full train')
            plt.axis([0, self.iter_thresh, 0, max(cost_vec)])
            plt.legend(loc='lower right', shadow=True)
            plt.show()
        else:
            return cost_vec
        
    def predict(self, data):
        m = data.shape[1]
        theta1 = self.theta[:((m+1) * self.nn_architecture[0])]
        theta1 = theta1.reshape(self.nn_architecture[0], m+1)
    
        theta2 = self.theta[nn.nn_architecture[0] * (m+1):]
        theta2 = theta2.reshape(nn.nn_architecture[1], nn.nn_architecture[0]+1)
    
        a1 = self.appendOnes(data)
        z2 = self.h_theta(a1, theta1, append_ones=False)
        a2 = self.appendOnes(nn.sigmoid(z2))
        z3 = self.h_theta(a2, theta2, append_ones=False)
        a3 = self.sigmoid(z3)
        
        out2    = np.zeros(a3.shape)
        indices = a3.argmax(axis=1)
        for i in range(indices.shape[0]):
            out2[i, indices[i]] = 1
        return out2.argmax(axis=1)
    
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

#    x=np.array([[1, 2, 3], [2, 4, 5], [8, 9, 10]])
#    y=np.array([[1], [2], [2]])

    archi = np.array([25, len(np.unique(y))])
    
    theta_size = archi[0] * (x.shape[1]+1)
    
    for i in range(archi.shape[0]):
        if i != 0:
            theta_size += archi[i] * (archi[i-1]+1)
    theta = np.random.rand(theta_size)
    alpha = 1 * np.ones(len(archi))
    lmbda = 100 * np.ones(len(archi))
    
    nn = NeuralNetwork2(archi, x, y, 0.2)
    nn.do_kfold_cross_validation()
    nn.train(x, y, append_ones=True, c_valid=False)
    pred = nn.predict(x)
    print 'Final Accuracy:', nn.accuracy(pred, y)*100
        