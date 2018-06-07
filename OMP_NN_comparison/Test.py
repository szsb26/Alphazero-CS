#Test Class loads the the Testing environment. It also provides methods for graph drawing, etc...
#Takes in models which have been set to True in main and compares them. For Neural networks set to True in args, Test class
#already assumes that these Neural Networks have been trained. 
import matplotlib.pyplot as plt
from Gen_Data import GenData
from NNet import NNetWrapper
import os
from sklearn.linear_model import orthogonal_mp
import numpy as np
import csv

class Test():
    def __init__(self, args):
        self.args = args
        self.Data = GenData()
        self.NNet = NNetWrapper(args) #NNetWrapper which holds 2 neural networks as self parameters
        
    def trainAll(self): #train over all m generated matrices 
    #INPUT:
    #OUTPUT: saved models and weights of neural networks for each matrix and their generated training signals. from GenData
        #For each matrix m, train respective Naive_net
        if self.args['naive_NN']:
            for m in range(self.args['matrices_generated']):
                if not os.path.exists(self.args['naiveNN_savedmodels_filepath'] + '/matrix' + str(m)):
                    os.mkdir(self.args['naiveNN_savedmodels_filepath'] + '/matrix' + str(m))
                self.NNet.naiveNN_train(self.args['naiveNN_training_filepath'] + '/matrix' + str(m), self.args['naiveNN_savedmodels_filepath'] + '/matrix' + str(m))    
                self.NNet.naive_net = self.NNet.netWrapper.naiveNet(self.args) #reinitialize self.NNet object
        #For each matrix m, train respective OMPbootstrapNN
        if self.args['OMPbootstrap_NN']:
            for m in range(self.args['matrices_generated']):
                if not os.path.exists(self.args['OMPbootstrap_savedmodels_filepath'] + '/matrix' + str(m)):
                    os.mkdir(self.args['OMPbootstrap_savedmodels_filepath'] + '/matrix' + str(m))
                self.NNet.OMPbootstrapNN_train(self.args['OMPbootstrap_training_filepath'] + '/matrix' + str(m), self.args['OMPbootstrap_savedmodels_filepath'] + '/matrix' + str(m))
                self.NNet.OMPbootstrap_net = self.NNet.netWrapper.OMPbootstrap_Net(self.args) #reinitialize self.OMPbootstrap_Net object. Note that .netWrapper is a NetArch object 
                
    def computeAccuracy(self): 
    #INPUT:
    #OUTPUT:A list of numpy vectors corresponding to each model, where each vector is of max size args['m'], and contains the model's accuracy on those set of test signals.
    #The first entry of such a vector corresponds to the accuracy on 1-sparse signals, the second entry corresponds to the accuracy on 2-sparse signals, etc...
    #FUNCTION: Compare algorithms on the data in filepath /testData/matrix('m')/('s')sparse/('s')sparse_obsy.csv
        accuracy_OMPbootstrap = np.zeros(self.args['m'])
        accuracy_OMP = np.zeros(self.args['m'])
    
        for m in range(self.args['matrices_generated']):
        
            #load the model
            filepath_model = self.args['OMPbootstrap_savedmodels_filepath'] + '/matrix' + str(m) + '/OMPbootstrapNN_model.json'
            filepath_weights = self.args['OMPbootstrap_savedmodels_filepath'] + '/matrix' + str(m) + '/OMPbootstrapNN_weights.h5'
            self.NNet.load_OMPbootstrapNN_model(filepath_model, filepath_weights)
            #load the matrix
            A = np.load(self.args['matrix_folder'] + '/matrix' + str(m) + '/' + self.args['matrix_type'] + str(m) + '.npy')
            #define what folder test signals are located for a given matrix
            test_signals_filepath = self.args['testData_folder'] + '/matrix' + str(m)
            #Initialize accuracy vector for given matrix
            acc_OMPbootstrap_matrix = np.zeros(self.args['m'])
            acc_OMP_matrix = np.zeros(self.args['m'])
            
            for s in range(1,self.args['max_sparsity']):
                obsy_filepath = test_signals_filepath + '/' + str(s) + 'sparse' + '/' + str(s) + 'sparse_obsy.csv'
                sparsex_filepath = test_signals_filepath + '/' + str(s) + 'sparse' + '/' + str(s) + 'sparse_x.csv'
                with open(obsy_filepath) as obsy, open(sparsex_filepath) as sparsex:
                    reader_y = csv.reader(obsy)
                    reader_x = csv.reader(sparsex)
                    counter_bootstrap = 0 #counter increases if predicted x is close to real x
                    counter_OMP = 0 #counter increases if predicted x is close to real x
                    signal_counter = 1 #count number of signals for printing below
                    for y, x in zip(reader_y, reader_x): 
                        #Convert read data into numpy arrays:
                        y = list(map(float, y))
                        y = np.asarray(y)
                        x = list(map(float, x))
                        x = np.asarray(x)
                        #Test Recovery statistics-------------------------------
                        x_bootstrap = self.NNet.OMPbootstrapNN_predict_x(A, y, s)
                        x_OMP = orthogonal_mp(A, y, n_nonzero_coefs = s)
                        
                        error_bootstrap = np.linalg.norm(x_bootstrap - x)**2
                        error_OMP = np.linalg.norm(x_OMP - x)**2
                        
                        print('')
                        print('CURRENT SIGNAL Y: ' + str(signal_counter) + '---------------------> ')
                        print('x_bootstrap: ' + str(x_bootstrap))
                        print('x_OMP: ' + str(x_OMP))
                        print('x: ' + str(x))
                        print('y: ' + str(y))
                        print('-------------------------------------------> ')
                        print('')
                        
                        if error_bootstrap <= self.args['epsilon']:
                            counter_bootstrap += 1/self.args['test_signals_generated']
                            
                        if error_OMP <= self.args['epsilon']:
                            counter_OMP += 1/self.args['test_signals_generated']
                        
                        signal_counter += 1    
                        #--------------------------------------------------------
                    
                #Update acc_OMPbootstrap_matrix and acc_OMP_matrix 
                acc_OMPbootstrap_matrix[s] = counter_bootstrap
                #print(acc_OMPbootstrap_matrix)
                acc_OMP_matrix[s] = counter_OMP   
                #print(acc_OMP_matrix)
            #update accuracy_OMPbootstrap and accuracy_OMP
            accuracy_OMPbootstrap += acc_OMPbootstrap_matrix/self.args['matrices_generated']
            accuracy_OMP += acc_OMP_matrix/self.args['matrices_generated']
        
        return accuracy_OMPbootstrap, accuracy_OMP