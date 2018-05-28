import sys
#Add Class files to the sys search path for importing later on
sys.path.insert(0,'/Users/sichenzhong/Desktop/Sichen/Graduate_School/ML/NN_MCTS_CS/python_src/alphazero_compressedsensing')
sys.path.insert(0,'/Users/sichenzhong/Desktop/Sichen/Graduate_School/ML/NN_MCTS_CS/python_src/alphazero_compressedsensing/compressed_sensing')
sys.path.insert(0,'/Users/sichenzhong/Desktop/Sichen/Graduate_School/ML/NN_MCTS_CS/python_src/alphazero_compressedsensing/compressed_sensing/keras_tf')
from CSGame import CSGame
from CSState import State
from Game_Args import Game_args
from NNet import NNetWrapper
from Coach import Coach
from MCTS import MCTS
import numpy as np

#Testing the MCTS with NN module. 
#Initialize for MCTS search

args = {
    #CS Parameters
    'matrix_type': 'sdnormal',
    'x_type': 'sdnormal',
    'm': 5,
    'n':15,
    'sparsity':5,
    #NN Parameters
    'lr':0.001,
    'num_layers':2,
    'neurons_per_layer':50,
    'epochs':10,
    'batch_size':64,
    'num_features':2,
    'x_l2': True,
    'lambda': True, 
    #MCTS parameters
    'cpuct': 1,
    'numMCTSSims':200,
    'tempThreshold':15,
    'gamma': 1,
    'epsilon': 1e-5,
}

#Initialize Game_args, NNetWrapper, CSGame, and MCTS
test_game_args = Game_args()
test_game_args.generateSensingMatrix(args['m'], args['n'], args['matrix_type'])
test_game_args.generateNewObsVec(args['x_type'], args['sparsity'])

test_nnet = NNetWrapper(args)
test_game = CSGame()
test_MCTS = MCTS(test_game, test_nnet, args, test_game_args)  

#TESTING STATE METHODS AND NN PREDICTIONS
#----------------------------------------------------

#Test and output state self variables
game_start = test_game.getInitBoard(args, test_game_args)
#print('The action indices are: ' + str(game_start.action_indices))
#print('')
#print('The column indices are: ' + str(game_start.col_indices))
#print('')

#Compute NN representation and test prediction
#game_start.compute_x_S_and_res(args, test_game_args)
#print('Feature Dictionary info: ')
#print(game_start.feature_dic)
#print('')
#game_start.converttoNNInput()
#print('NN_input data info: ')
#print(game_start.nn_input)
#print(len(game_start.nn_input))
#print(game_start.nn_input[0].shape)
#print(game_start.nn_input[1].shape)
#print('')

#TESTING SOME NNETWRAPPER METHODS
#----------------------------------------------------
#p_as, z = test_nnet.predict(game_start)
#print('The predicted probability dist is: ')
#print(p_as)
#print(p_as.shape)
#print('')
#print('The predicted reward is: ')
#print(z)
#print(z.shape)
#print('')


#TESTING MCTS OBJECT AND ITS METHODS
#----------------------------------------------------
print('Starting MCTS search on just the initial game state...')
print('')
print('The statistics of the initial game state are:')
print(game_start.action_indices)
print(game_start.col_indices)
probs = test_MCTS.getActionProb(game_start) #50 numMCTSSims have been run to find the next move
print('')
print('Starting from the initial game state, the prob. dist. of the next move is:')
print(probs)
print('')

#print('After searching on the initial game state, the current self variables of the MCTS class are:')
#print('self.Qsa : ' + str(test_MCTS.Qsa))
#print('self.Nsa : ' + str(test_MCTS.Nsa))
#print('self.Ns : ' + str(test_MCTS.Ns))
#print('self.Ps : ' + str(test_MCTS.Ps))
#print('self.Es : ' + str(test_MCTS.Es))
#print('self.Vs : ' + str(test_MCTS.Vs))




