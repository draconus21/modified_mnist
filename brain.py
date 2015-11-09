# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:25:59 2015

@author: neeth
"""

import numpy as np
import matplotlib.pyplot as plt
#import abc as ABCMeta

class Brain(object):
    """Interface for learning."""

    
    def __init__(self, train_X, train_Y, c_ratio):
        '''Initializes learner object
        All prediction variables should be initialized in 
        derived class's __init__()
        The following is done here. Any other pre-processsing specific
        to dervied classes should be done in the derived class
        
        1. sets cross-validation partitions.
        '''
        
        self.x = np.array(train_X)
        self.y = np.array(train_Y)
        
        # orient y matrix correctly
#        if self.y.shape[1] != 1 or self.y.shape[1] > self.y.shape[0]:
#            self.y = self.y.T
        
        self.n = self.x.shape[0]
        self.m = self.x.shape[1]
        
        self.train_len = 0
        self.cross_len = 0
        self.c_ratio   = c_ratio
            
        self.train_X   = []
        self.train_Y   = []
        self.c_valid_X = []
        self.c_valid_Y = []
    
        self.c_indices = []

        self.set_cross_validation_sets()
        
    def set_cross_validation_sets(self):
        '''Decides the partitions for different cross-validation sets
        and populates c_indices.'''
        
        if self.c_ratio > 1 or self.c_ratio <= 0:
            print 'invalid c_ratio:', self.c_ratio
            self.c_indices = np.array([-1, -1])
            return
        
        block_size = (int)(self.n * self.c_ratio)
        
        if block_size < 1:
            print 'cross-validation block_size is less than 1:', block_size       
            self.c_indices = np.array([-1, -1])
            return
        
        n_blocks = (int)(self.n / block_size)
        
        if self.n % block_size != 0:
            n_blocks += 1
        
        self.c_indices = np.empty([n_blocks, 2])
        
        self.c_indices[:, 0] = list(range(0, n_blocks * block_size, block_size))
        self.c_indices[:, 1] = list(range(block_size, n_blocks * block_size + 1, block_size))
        
        self.c_indices[-1, 1] = self.n  
            
    def set_data(self, c_index):
        '''Populates train_X, train_Y, c_valid_X and c_valid_Y 
        for cross validation round i.'''
        
        if np.any(self.c_indices == -1):
            self.train_len = self.n
            self.cross_len = 0
            
            self.train_X = self.x
            self.train_Y = self.y
            
            self.c_valid_X = np.array([0, 0])
            self.c_valid_Y = np.array([0, 0])
        else:
            lower = self.c_indices[c_index][0]
            upper = self.c_indices[c_index][1]
            
            self.cross_len = upper - lower
            self.train_len = self.n - self.cross_len
            
            self.c_valid_X = self.x[lower:upper, :]
            self.c_valid_Y = self.y[lower:upper]
            
            self.train_X   = np.empty([self.train_len, self.m])
            self.train_Y   = np.empty([self.train_len])
            
            l_index = 0
            if lower-1>= 0:
                self.train_X[:lower, :] = self.x[:lower, :]
                self.train_Y[:lower] = self.y[:lower]
                l_index = lower
            if upper < self.n:
                self.train_X[l_index:, :] = self.x[upper:self.n, :]
                self.train_Y[l_index:] = self.y[upper:self.n]
            
    def do_kfold_cross_validation(self):
        iter = list(range(self.c_indices.shape[0]))
        cross_error   = 0
        cross_err_mat = []
        #Itertatively perform cross validation
        for i in iter:
            cost_vec = []
            self.set_data(i)
            cost_vec = self.train(self.train_X, self.train_Y, c_valid = True)
            if self.cross_len > 0:
                predict = self.predict(self.c_valid_X)
                c_err = (1-self.accuracy(predict, self.c_valid_Y)) * 100
                cross_err_mat = np.append(cross_err_mat, c_err)
                print 'cross_error', i, ':', c_err
                print 'accuracy   ', i, ':', self.accuracy(predict, self.c_valid_Y) * 100
                cross_error += c_err ** 2
            plt.plot(cost_vec, label = 'cross-'+str(i))
            plt.axis([0, self.iter_thresh, 0, max(cost_vec)])
            plt.legend(loc='lower right', shadow=True)            
            plt.show()
            
        cross_error /= iter[-1]
        plt.plot(cross_err_mat)
        plt.axis([0, iter[-1], 0, np.max(cross_err_mat)])
        plt.show()
        return cross_error, cross_err_mat

#    @abstractmethod
    def train(self, x, y, c_valid = False):
        '''Trains the data
        x      : (nxm array) Training data
        y      : (nx1 array) Target data
        c_valid: (Boolean) Set to true when learning for cross_validation
        
        Does not return any value. Learned parameters are stored within the class
        Note: n is the number of examples in training data set, m is the 
              number of features.'''
    
#    @abstractmethod
    def calc_error(self, predict, y):
        '''Calculates error between predicted values and actual values
        predict : (nx1 array) Predicted values
        y      : (nx1 array) Actual Values
        
        @return: (float) error
        Note: n is the number of examples.'''
    
#    @abstractmethod
    def predict(self, x, c_valid = False):
        '''Make predictions based on trained model
        x      : (nxm array) Data to be used for prediction
        c_valid: (Boolean) Set to true when learning for cross_validation
        
        @return: (nx1 array) Predicted values
        Note: n is the number of examples in training data set.'''
    
    def tester(self):
        '''This is just for testing
        Add the code to be tested into this function.
        This will be called from main().
        Remove 'pass' when there is code added.'''
#        self.do_kfold_cross_validation()
        pass

def accuracy(predict, y):
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
    temp_y = np.sort(y)
    temp_p = np.sort(predict)
    unique_y, count_y       = np.unique(temp_y, return_counts=True)
    unique_pred, count_pred = np.unique(temp_p, return_counts=True)
    if len(unique_y) != len(unique_pred):
        print 'not equal:', len(unique_y), len(unique_pred)
        print unique_y
        print unique_pred
        
    print unique_y.shape, unique_pred.shape
        
    for i in range(len(unique_y)):
        print unique_y[i], ': ', count_y[i], '| ', unique_pred[i], ': ', count_pred[i]
    print 'Accuracy %d / %d = %.4f' % (corr, len(y), acc)

if __name__=='__main__':
    x = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    n = np.array([1, 0, 1, 0]).T
   
    