#Load each model and weight and corresponding training set to compute loss

import matplotlib
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.optimizers import Adam
import numpy as np
import pickle
from pickle import Pickler, Unpickler
import os
import sys
from random import shuffle
sys.path.insert(0,'/Users/sichenzhong/Desktop/Sichen/Graduate_School/ML/NN_MCTS_CS/python_src/alphazero_compressedsensing_nonoise_nobootstrap/compressed_sensing/keras_tf')
sys.path.insert(0,'/Users/sichenzhong/Desktop/Sichen/Graduate_School/ML/NN_MCTS_CS/python_src/alphazero_compressedsensing_nonoise_nobootstrap/compressed_sensing')
sys.path.insert(0,'/Users/sichenzhong/Desktop/Sichen/Graduate_School/ML/NN_MCTS_CS/python_src/alphazero_compressedsensing_nonoise_nobootstrap')
from CSState import State
from CSGame import CSGame
from Game_Args import Game_args
from NNet import NNetWrapper
from Coach import Coach

#--------------------------------------------------------------------------------------
def constructTraining(states): #this method is used in Coach.py
#INPUT: a list of state objects which have values for self.feature_dic, self.p_as, and self.z
#OUTPUT: A list [X,Y] training data.
    num_states = len(states)
    X = []
    Y = []
    #Initialize every entry in X as an empty numpy array matrix     
    for key in states[0].feature_dic: #Each state's feature dictionary should contain vectors which are all the same size
        zero_feature_matrix = np.empty((num_states,len(states[0].feature_dic[key])))
        X.append(zero_feature_matrix)
    
    #Fill in the rows of each empty matrix in list X
    for i in range(num_states): #iterate over the number of states, which is equal to the row dim of every np array in X.
        list_index = 0
        for key in states[i].feature_dic:
            X[list_index][i][:] = states[i].feature_dic[key] 
            list_index += 1 
                
    #Construct labels Y, which is length 2 list of numpy arrays
    pi_as_empty = np.empty((num_states, states[0].action_indices.size))
    z_empty = np.empty((num_states,1))
    Y.append(pi_as_empty)
    Y.append(z_empty)
            
    for i in range(num_states): #Y[0].shape equals number of states
        Y[0][i][:] = states[i].pi_as
        Y[1][i][0] = states[i].z
        
    converted_training = [X, Y]
    return converted_training
#--------------------------------------------------------------------------------------


part1 = False
part2 = False
part3 = True


alphazero_iterations = 198
#Compute the loss of every NN on their respective training sets
if part1:
    #initialize training_loss vector
    training_loss = np.zeros(alphazero_iterations)
    #open up training_files and load_models for prediction
    for i in range(alphazero_iterations):
        #load json and create model
        modelfilepath = os.getcwd() + '/network_checkpoint/nnet_checkpoint' + str(i) + '_model.json'
        json_file = open(modelfilepath, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        #load weights into new model
        weightsfilepath = os.getcwd() + '/network_checkpoint/nnet_checkpoint' + str(i) + '_weights.h5'
        loaded_model.load_weights(weightsfilepath)
        print('Loaded model from disk')
        #Evaluate model on training set to compute loss
        loaded_model.compile(loss = ['categorical_crossentropy', 'mean_squared_error'], metrics = ['accuracy'], optimizer = Adam(0.001))
        #Load the training data so we can evaluate model on training data
        trainExamplesfile = os.getcwd() + '/training_data/checkpoint_' + str(i) + '.pth.tar.examples'
        with open(trainExamplesfile, 'rb') as f:
            trainExamplesHistory = Unpickler(f).load()
        f.closed
        trainExamples = []
        #convert trainExamplesHistory into a list
        for e in trainExamplesHistory: #Each e is a deque
            trainExamples.extend(e)
            shuffle(trainExamples)
        #convert trainExamples into format NN recognizes
        trainExamples = constructTraining(trainExamples)
        #evaluate
        evaluation = loaded_model.evaluate(trainExamples[0], trainExamples[1], verbose = 0)
        print(loaded_model.metrics_names)
        print('Training Loss:', evaluation[0])
        print('Training accuracy:', evaluation[3])
        training_loss[i] = evaluation[0]
    
    #save the training loss list via pickling
    with open("training_loss.txt", "wb") as data:
        pickle.dump(training_loss, data)

#plot the training loss
if part2:
    with open('training_loss.txt', 'rb') as data:
        training_loss = pickle.load(data)

    plt.plot(training_loss)
    plt.legend(['train'], loc='upper left')
    plt.title('Alphazero Training Loss to Iterations')
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.savefig('training_loss.png')

#Compute the validation loss of every NN on the single test set
if part3:
    args = {
    #Compressed Sensing Parameters, Ax = y, where A is of size m by n
    'fixed_matrix': True, #fixes a single matrix across entire alphazero algorithm. If set to True, then self play games generated in each iteration have different sensing matrices. The below options will not run if this is set to False.
        'load_existing_matrix': True, #If we are using a fixed_matrix, then this option toggles whether to load an existing matrix from args['fixed_matrix_filepath'] or generate a new one. If loading an existing matrix, the matrix must be saved as name 'sensing_matrix.npy'
            'matrix_type': 'sdnormal',  #type of random matrix generated if(assuming we are not loading existing matrix)
    'x_type': 'uniform01',  #type of entries generated for sparse vector x
    'm': 7, #row dimension of A
    'n':15, #column dimension of A
    'sparsity':7, #dictates the maximum sparsity of x when generating the random vector x. Largest number of nonzeros of x is sparsity-1. sparsity cannot be greater than m above. 
    'fixed_matrix_filepath': os.getcwd() + '/fixed_sensing_matrix', #If args['fixed_matrix'] above is set to True, then this parameter determines where the fixed sensing matrix is saved or where the existing matrix is loaded from. 
    #---------------------------------------------------------------
    #General Alphazero Training Parameters
    'numIters': 500, #number of alphazero iterations performed. Each iteration consists of 1)playing numEps self play games, 2) retraining neural network
    'numEps': 400, #dictates how many self play games are played each iteration of the algorithm
    'maxlenOfQueue':10000, #dictates total number of game states saved(not games). 
    'numItersForTrainExamplesHistory': 1, #controls the size of trainExamplesHistory, which is a list of different iterationTrainExamples deques. 
    'checkpoint': os.getcwd() + '/training_data', #filepath for SAVING newly generated self play training data
    'load_training': True, #If set to True, then load latest batch of self play games for training. 
        'load_folder_(folder)': os.getcwd() + '/training_data', #filepath for LOADING the latest set of training data
        'load_folder_(filename)': 'best.pth.tar', #filename for LOADING the latest generated set of training data. Currently, this must be saved as 'best.pth.tar'
    'Arena': False, #determines whether model selection/arena is activated or not. Below options will not be run if this is set to False.
        'arenaCompare': 100, #number of games played in the arena to compare 2 networks pmcts and nmcts
        'updateThreshold': 0.55, #determines the percentage of games nmcts must win for us to update pmcts to nmcts
    #---------------------------------------------------------------
    #NN Parameters
    'lr': 0.001,    #learning rate of NN
    'num_layers': 2,    #number of hidden layers after the 1st hidden layer
    'neurons_per_layer':200,    #number of neurons per hidden layer
    'epochs': 10,   #number of training epochs. If There are K self play states, then epochs is roughly K/batch_size. Note further that K <= numEps*sparsity. epochs determines the number of times weights are updated.
    'batch_size': 400, #dictates the batch_size when training 
    'num_features' : 2, #number of self-designed features used in the input
    'load_nn_model' : True, #If set to True, load the best network (best_model.json and best_weights.h5)
    'network_checkpoint' : os.getcwd() + '/network_checkpoint', #filepath for SAVING the temp neural network model/weights, checkpoint networks model/weights, and the best networks model/weights
    #features: True if we wish to use as a feature, False if we do not wish to use as a feature
    'x_l2' : True,      #solution to min_z||A_Sz - y||_2^2, where A_S is the submatrix of columns we have currently chosen
    'lambda' : True,    #the vector of residuals, lambda = A^T(A_Sx-y), where x is the optimal solution to min_z||A_Sz - y||_2^2
    #---------------------------------------------------------------
    #MCTS parameters
    'cpuct': 1, #controls the amount of exploration at each depth of MCTS tree.
    'numMCTSSims': 500, #For each move, numMCTSSims is equal to the number of MCTS simulations in finding the next move during self play. 
    'tempThreshold': 0,    #dictates when the MCTS starts returning deterministic polices (vector of 0 and 1's). See Coach.py for more details.
    'gamma': 1, #note that reward for a terminal state is -alpha||x||_0 - gamma*||A_S*x-y||_2^2. The smaller gamma is, the more likely algorithm is going to choose stopping action earlier(when ||x||_0 is small). gamma enforces how much we want to enforce Ax is close to y. We need gamma large enough!!!
    'alpha': 1e-5, #note that reward for a terminal state is -alpha||x||_0 - gamma*||A_S*x-y||_2^2. The smaller alpha is, the more weight the algorithm gives in selecting a sparse solution. 
    'epsilon': 1e-5, #If x is the optimal solution to l2, and the residual of l2 regression ||A_Sx-y||_2^2 is less than epsilon, then the state corresponding to indices S is a terminal state in MCTS. 
    }
    
    test_loss = np.zeros(alphazero_iterations)
    
    for i in range(alphazero_iterations):
        #-------------------------------------------------------------
        game_args = Game_args()
        matrix_filename = 'sdnormal0.npy'
        A = np.load(matrix_filename)
        game_args.sensing_matrix = A
        nnet = NNetWrapper(args)
        model_filepath = os.getcwd() + '/network_checkpoint'
        model_filename = 'nnet_checkpoint' + str(i)
        nnet.load_checkpoint(model_filepath, model_filename)
        new_game = CSGame()
        Alphazero = Coach(new_game, nnet, args, game_args)
        #--------------------------------------------------------------
        #Alphazero now ready for prediction
        avgloss_for_NN = 0
    
