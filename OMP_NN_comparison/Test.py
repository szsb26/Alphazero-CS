#Test Class loads the the Testing environment. It also provides methods for graph drawing, etc...
#Takes in models which have been set to True in main and compares them. For Neural networks set to True in args, Test class
#already assumes that these Neural Networks have been trained. 
import matplotlib.pyplot as plt
from Gen_Data import GenData
from NNet import NNetWrapper
import os

class Test():
    def __init__(self, args):
        self.args = args
        self.Data = GenData()
        self.NNet = NNetWrapper(args)
        
    def trainAll(self): #train over all m generated matrices 
    #INPUT:
    #OUTPUT: saved models and weights of neural networks for each matrix and their generated training signals. from GenData
        for m in range(self.args['matrices_generated']):
            if self.args['naive_NN'] == True:
                os.mkdir(self.args['naiveNN_savedmodels_filepath'] + '/matrix' + str(m))
                self.NNet.naiveNN_train(self.args['naiveNN_training_filepath'] + '/matrix' + str(m), self.args['naiveNN_savedmodels_filepath'] + '/matrix' + str(m))    
                self.NNet.naive_net = self.NNet.netWrapper.naiveNet(self.args) #reinitialize self.NNet object
            if self.args['OMPbootstrap_NN'] == True:
                pass
                
    def load_naiveNet_model(self, filepath):
    #INPUT: None
    #OUTPUT: None
    #FUNCTION:Load the model corresponding to filepath. Namely, load neural net model and weights into self.NNet.naive_net for prediction.
        pass
    def load_OMPbootstrapNet_model(self, filepath):
    #INPUT: None
    #OUTPUT: None
    #FUNCTION:Load the model corresponding to filepath. Namely, load neural net model and weights into self.NNet.OMPbootstrap_net for prediction
        pass
    def compute_accuracy(self):
    #INPUT: None
    #OUTPUT: None
    #FUNCTION: Using the algorithms set to true in args, test the accuracy on generated test signals for each sparsity.
        pass