import multiprocessing
import numpy as np
import sys

sys.path.append("..")
sys.path.append("../compressed_sensing")
sys.path.append("../compressed_sensing/keras_tf")

from CSState import State
from CSGame import CSGame
from MCTS import MCTS
from Game_Args import Game_args
from NNet import NNetWrapper
from collections import deque
from Arena import Arena
from batch_MCTS import batch_MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import multiprocessing


args = {
    #Compressed Sensing Parameters, Ax = y, where A is of size m by n
    'fixed_matrix': True, #fixes a single matrix across entire alphazero algorithm. If set to False, then self play games generated in each iteration have different sensing matrices. The below options will not run if this is set to False.
        'load_existing_matrix': True, #If we are using a fixed_matrix, then this option toggles whether to load an existing matrix from args['fixed_matrix_filepath'] or generate a new one. If loading an existing matrix, the matrix must be saved as name 'sensing_matrix.npy'
            'matrix_type': 'sdnormal',  #type of random matrix generated if(assuming we are not loading existing matrix)
    'x_type': 'uniform01',  #type of entries generated for sparse vector x when playing games of self-play
    'm': 7, #row dimension of A
    'n': 15, #column dimension of A
    'sparsity':7, #dictates the maximum sparsity of x when generating the random vector x. Largest number of nonzeros of x is sparsity-1. sparsity cannot be greater than m above. 
    'fixed_matrix_filepath': os.getcwd() + '/fixed_sensing_matrix', #If args['fixed_matrix'] above is set to True, then this parameter determines where the fixed sensing matrix is saved or where the existing matrix is loaded from. 
    #---------------------------------------------------------------
    #General Alphazero Training Parameters
    'num_batches': 1, #determines how many batches of (y,x) pairs we wish to generate. 
    'eps_per_batch': 5, #number of MCTS objects/(y,x) pairs we want to maintain at once for parallel search. Note that each MCTS tree is saved in memory, so this option should not exceed total memory.
    'numIters': 100, #number of alphazero iterations performed. Each iteration consists of 1)playing numEps self play games, 2) retraining neural network
    'maxlenOfQueue':10000, #dictates total number of game states saved(not games). 
    'numItersForTrainExamplesHistory': 1, #controls the size of trainExamplesHistory, which is a list of different iterationTrainExamples deques. 
    'checkpoint': os.getcwd() + '/training_data', #filepath for SAVING newly generated self play training data
    'load_training': False, #If set to True, then load latest batch of self play games for training. 
        'load_folder_(folder)': os.getcwd() + '/training_data', #filepath for LOADING the latest set of training data
        'load_folder_(filename)': 'best.pth.tar', #filename for LOADING the latest generated set of training data. Currently, this must be saved as 'best.pth.tar'
    'Arena': False, #determines whether model selection/arena is activated or not. Below options will not be run if this is set to False.
        'arenaCompare': 100, #number of games played in the arena to compare 2 networks pmcts and nmcts
        'updateThreshold': 0.55, #determines the percentage of games nmcts must win for us to update pmcts to nmcts
    'verbose': True, #determines whether comments are enabled or not. 
    'processes': 6, #determines multiprocessing.pool(x)
    #---------------------------------------------------------------
    #NN Parameters
    'lr': 0.001,    #learning rate of NN, relevant for NetArch(), NetArch1()
    'num_layers': 2,    #number of hidden layers after the 1st hidden layer, only relevant for NetArch()
    'neurons_per_layer': 200,    #number of neurons per hidden layer
    'epochs': 20,   #number of training epochs. If There are K self play states, then epochs is roughly K/batch_size. Note further that K <= numEps*sparsity. epochs determines the number of times weights are updated.
    'batch_size': 400, #dictates the batch_size when training 
    'num_features' : 2, #number of self-designed features used in the input
    'load_nn_model' : False, #If set to True, load the best network (best_model.json and best_weights.h5)
    'network_checkpoint' : os.getcwd() + '/network_checkpoint', #filepath for SAVING the temp neural network model/weights, checkpoint networks model/weights, and the best networks model/weights
    #features: True if we wish to use as a feature, False if we do not wish to use as a feature
    'x_l2' : True,      #solution to min_z||A_Sz - y||_2^2, where A_S is the submatrix of columns we have currently chosen
    'lambda' : True,    #the vector of residuals, lambda = A^T(A_Sx-y), where x is the optimal solution to min_z||A_Sz - y||_2^2
    #---------------------------------------------------------------
    #MCTS parameters
    'cpuct': 2, #controls the amount of exploration at each depth of MCTS tree.
    'numMCTSSims': 500, #For each move, numMCTSSims is equal to the number of MCTS simulations in finding the next move during self play. Smallest value of numMCTSsims is 2.    
    'beta': 1, #Recall the augmented probability aug_prob = beta * probs + (1-beta) * 1/(len(x)) * x_I, where x_I is the indicator vector of ones of the true sparse solution x. Hence, higher beta values increase the probabilities towards choosing the correct column choices. 
                 #SET beta = 1 DURING TESTING SINCE x SHOULD BE UNKNOWN DURING TESTING. 
    'tempThreshold': 10,    #dictates when the MCTS starts returning deterministic polices (vector of 0 and 1's). See Coach.py for more details.
    'alpha': 1e-5,  #note that reward for a terminal state is -alpha||x||_0 - gamma*||A_S*x-y||_2^2. The smaller alpha is, the more weight the algorithm gives in selecting a sparse solution.
    'gamma': 1, #note that reward for a terminal state is -alpha||x||_0 - gamma*||A_S*x-y||_2^2. The smaller gamma is, the more likely algorithm is going to choose stopping action earlier(when ||x||_0 is small). gamma enforces how much we want to enforce Ax is close to y. 
                #choice of gamma is heavily dependent on the distribution of our signal and the distribution of entries of A. gamma should be apx. bigger than m/||A_Sx^* - y||_2^2, where y = Ax, and x^* is the solution to the l2 regression problem.
    'epsilon': 1e-5, #If x is the optimal solution to l2, and the residual of l2 regression ||A_Sx-y||_2^2 is less than epsilon, then the state corresponding to indices S is a terminal state in MCTS. 
}

 
    
class Test():
    def __init__(self, game, args, game_args):
        self.args = args
        self.game_args = game_args
        self.arena_game_args = Game_args()
        self.game = game
        #self.nnet = nnet #UNCOMMENTING THIS LINE CAUSES THREADLOCK ERROR "can't pickle _thread.lock objects". Seems like its a conflict of pickling the nnet object.
                         #When nnet = NNetWrapper(args) is instantiated, two self variables are automatically constructed, namely self.args and self.nnet = NetArch2(args).
                         #self.args causes us no trouble(due to tests), but self.nnet = NetArch2(args) does. I'm guessing the problem here is that when
                         #Test.do_nothing(self, num) is split across multiple processes, the self variables of Test are also pickled(and copied to 
                         #each process?), but due to the fact that NetArch2 is a class object (with self.args, self.p_as, self.v, and self.model) class variables, 
                         #the self.model self variable is causing problems in that it is not able to be pickled.
                         
                         #Line 101 in CS_NNet.py is the specific line which causes our pickling error, namely
                         # self.p_as = Dense(self.args['n']+1, activation = 'softmax', name = 'p_as')(hidden_layer)
        
        #self.pnet = self.nnet.__class__(self.args)
        #self.batch_mcts = batch_MCTS(args, nnet) #UNCOMMENTING THIS LINE CAUSES THREADLOCK ERROR "can't pickle _thread.lock objects"
        #self.trainExamplesHistory = []
        #self.skipFirstSelfPlay = False
        
    def State_op(self, State_list):
        for first_State, sec_State in State_list:
            print(first_State.action_indices, sec_State.action_indices)
        return 0
    
    def test_func(self):
        p = multiprocessing.Pool(6)
        #Input must be in a list. First element of list is first call of State_op, second element is second call
        #of State_op, etc... In our example, State_list is our only input and we only call State_op once.
        Result = p.map(self.State_op, [State_list]) 
        print(Result)
        
        return 'Done'
        
    def do_nothing(self, num, nnet):
        print('I got here')
        return num

        
    def learn(self, nnet):
    
        p = multiprocessing.Pool(3)
        result = p.map(self.do_nothing, [nnet])
        print('result:', result)
        print('---------------------------Got Here------------------------------')
        return 0

class Alpha_NNet():
    def __init__(self, args, nnet):
        self.args = args
        self.nnet = nnet


nnet = NNetWrapper(args)
Alpha_NNet = Alpha_NNet(args, nnet)
Game_rules = Game_args()
Game = CSGame()
Test_obj = Test(Game, args, Game_rules)
Test_obj.learn(Alpha_NNet.nnet)





        