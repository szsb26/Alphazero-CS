#Class which generates sensing_matrices A, observed vectors y, sparse vectors x
import numpy as np
import math
from scipy.linalg import dft
import random
import csv
import os

class GenData(): #Generates A, x, y. Also generates training data for neural network models with different inputs. 
    def __init__(self):
        self.A = None
        self.y = None
        self.x = None
        
    def generate_SensingMatrix(self, args):
        if args['matrix_type'] == 'sdnormal':
            self.A = np.random.normal(0,1, (args['m'], args['n']))
        if args['matrix_type'] == 'fourier':
            DFT = dft(args['n'])
            rows = [x for x in range(args['m'])]
            random.shuffle(rows)
            self.A = DFT[np.ix_(rows)]
        
    def generate_ObsVector(self, args, sparsity): #generate y from a sparse vector x with given sparsity
        x = np.zeros(args['n'])
        if args['x_type'] == 'sdnormal':
            x[0:sparsity] = np.random.normal(0,1)
            np.random.shuffle(x)
            self.x = x
            self.y = np.matmul(self.A, self.x)
            
    def save_SensingMatrix(self, matrix_folder, matrix_filename): #matrix_folder is usually args['matrix_folder'].
        np.save(matrix_folder + matrix_filename, self.A)
        
    def compute_xS_colresIP(self):
        pass
    
    def gen_CSData(self, args):  #generate matrices, sparse vectors, and observed vectors and dump them into a file. Training data will be generated from these matrices, sparse vectors, and observed vectors using methods below.
        for m in range(args['matrices_generated']):
            self.generate_SensingMatrix(args)
            self.save_SensingMatrix(args['matrix_folder'], '/sensing_matrix' + str(m) + '.npy') 
            for s in range(1,args['max_sparsity']):
                Y = []
                X = []
                if not os.path.exists(args['matrix_folder'] + '/y_x_data' + '/' + str(s) + 'sparse'):
                    os.makedirs(args['matrix_folder'] + '/y_x_data' + '/' + str(s) + 'sparse')
                for num_signals in range(args['x_generated']):
                    self.generate_ObsVector(args, s)
                    Y.append(self.y)
                    X.append(self.x)
                with open(args['matrix_folder'] + '/y_x_data' + '/' + str(s) + 'sparse' + '/' + 'features_y.csv', 'a') as output: #'a' is append mode and will not overwrite file
                    writer = csv.writer(output)
                    writer.writerows(Y)
                with open(args['matrix_folder'] + '/y_x_data' + '/' + str(s) + 'sparse' + '/' + 'labels_x.csv', 'a') as output: #'a' is append mode and will not overwrite file
                    writer = csv.writer(output)
                    writer.writerows(X)
    
    def gen_naiveNNTraining(self, args): #self.gen_CSData(args) must be run first. Read in raw data and transform into something naive Net recognizes. Dump all this data into a new file. 
        for folder in os.listdir(args['matrix_folder'] + '/y_x_data/'):
            if not folder.startswith('.'): #ignore hidden files in directory
                print(str(folder))
    
    def gen_OMPbootstrapNNTraining(self, args): #self.gen_CSData(args) must be run first. Read in raw data and transform into something recognizable by OMPbootstrap NN. Needs to call compute_xS_colresIP above.
        pass
    
           

        
        

