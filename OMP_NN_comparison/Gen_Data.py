#Class which generates sensing_matrices A, observed vectors y, sparse vectors x
import numpy as np
import math
from scipy.linalg import dft
import random
import csv

class GenData(): #Generates A, x, y. Also generates training data for neural network models with different inputs. 
    def __init__(self):
        self.A = None
        self.y = None
        self.x = None
        
    def generate_SensingMatrix(self, args):
        if args['matrix_type'] == 'sdnormal':
            self.A = np.random.normal(0,1, (args['m'], args['n']))
        if args9['matrix_type'] == 'fourier':
            DFT = dft(n)
            rows = [x for x in range(args['m'])]
            random.shuffle(rows)
            self.A = DFT[numpy.ix_(rows)]
        
    def generate_ObsVector(self, args, sparsity): #generate y from a sparse vector x with given sparsity
        x = np.zeros(self.args['n'])
        if args['x_type'] == 'sdnormal':
            x[0:sparsity] = np.random.normal(0,1)
            np.random.shuffle(x)
            self.x = x
            self.y = np.matmul(self.A, self.x)
    
    def compute_xS_and_res(self.A, self.x, self.y, s): #This is used to compute the feature vectors and construct training data and labels for OMPbootstrap__NN below
    
            
    def generate_NNTraining(self, args): #generates training and their labels for neural networks if corresponding model in args is set to True
        #Initialize [Y,X] for naive NN. Y and X are directly fed into the neural network
        X = []
        Y = []
        
        #Initialize training and labels for OMPbootstrap_NN
        
        for m in range(args['matrices_generated']):
            self.generate_SensingMatrix(self, args)
            self.save_SensingMatrix(args['matrix_folder'], '/sensing_matrix' + str(m) + '.npy')
            for s in range(args['max_sparsity']):
                for signal in range(args['x_generated']):
                    self.generate_ObsVector(args, s)
                    
                    if args['naive_NN'] == True: #Construct training pairs in the form of (y,x)
                        X.append(self.x)
                        Y.append(self.y)
            
                    if args['OMPbootstrap_NN'] == True: 
                    #Need to call CSAlgorithms here (OMP) to generate each training,label pair. A training sample here is in the form of [(x_l2, col_res_IP), (01 vector of next column taken)] 
                    #Since s, x, A, and y are all fixed, we can now use OMP algorithm to generate a set of training samples for the previously fixed variables.
                    #Look at alphazero algorithm to show how these features are converted into input
                        for key in args['feature_dic']:
                            if args['feature_dic'][key] == True:
                            
                        
       
    def save_SensingMatrix(self, matrix_folder, matrix_filename): #matrix_folder is usually args['matrix_folder'].
        np.save(matrix_folder + matrix_filename, self.A)
           
#The below values are gotten from Donoho Tanner Figure 7 for m/n = 0.5 (magenta curve)
#One observes that recovery fails for matrices of size ratio 0.5 around sparsity/m = 0.45
#For sparsity/m greater than or equal to 0.45, one observes l1 recovery succeeds only 
#10 percent of the time
#n = 100
#m = 50
#sparsity = 0.45*m
#sensing_matrix = numpy.random.normal(0,1, (m, n))
#numpy.save('sensing_matrix.npy', sensing_matrix)


# Generate a Fourier sensing matrix
#n = 100
#m = 50
#sparsity = 0.45*m

#construct subsampled Fourier Matrix
#DFT = dft(n)
#rows = [x for x in range(m)]
#random.shuffle(rows)
#sampling_matrix = DFT[numpy.ix_(rows)]
#numpy.save('fourier_sensing_matrix.npy', sampling_matrix)
        
        
        

