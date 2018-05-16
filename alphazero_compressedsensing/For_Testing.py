import sys
#Add Class files to the sys search path for importing later on
sys.path.insert(0,'/Users/sichenzhong/Desktop/Sichen/Graduate School/ML/NN_MCTS_CS/python src/alphazero_compressedsensing/compressed_sensing')
sys.path.insert(0,'/Users/sichenzhong/Desktop/Sichen/Graduate School/ML/NN_MCTS_CS/python src/alphazero_compressedsensing/compressed_sensing')
sys.path.insert(0,'/Users/sichenzhong/Desktop/Sichen/Graduate School/ML/NN_MCTS_CS/python src/alphazero_compressedsensing/compressed_sensing/keras')
from CSGame import CSGame
from CSState import State
from NNet import NNetWrapper
import numpy as np


#Test the CSGame class
n = 10
m = 5
k = 3

A = np.random.normal(0,1, (m, n))
x = np.zeros(n)
x[:k] = np.random.uniform(0,1,k)
#np.random.shuffle(x)
y = np.matmul(A, x)

sample_game = CSGame(A, y, 4)

initial_state = sample_game.getInitBoard()
print('The Initial State is: ' + str(initial_state.action_indices))

action_size = sample_game.getActionSize()
print('Total action size is: ' + str(action_size))

state_actions_ind = np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1]) #size is n+1 due to stopping action
state_actions_ind2 = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1])

sample_state = State(state_actions_ind)
print('Action Indicator matrix for sample state is:' + str(sample_state.action_indices))

sample_state2 = State(state_actions_ind2)

action = 2
next_state = sample_game.getNextState(sample_state, action)
next_state.action_indices
print('The next state given action ' + str(action) + ' is: ' + str(next_state.action_indices))

valid_moves = sample_game.getValidMoves(sample_state)
print('The valid moves given as a list are:' + str(3) + ' is:' + str(valid_moves))

reward = sample_game.getGameEnded(sample_state)
print ('The reward for the current state is: ' + str(reward))
print('')

#Test the NNetWrapper class
args = {
    'lr': 0.001,
    'num_layers': 2,
    'neurons_per_layer':100,
    'epochs': 10,
    'batch_size': 2,
    'num_channels': 512,
    'num_features' : 2,
}
sample_nnetwrapper = NNetWrapper(args, sample_game)
single_training_sample = sample_nnetwrapper.statetoFeatures(sample_state, sample_game)
train_X = sample_nnetwrapper.convertStates([sample_state, sample_state2], sample_game)

print(train_X[0])
print(train_X[0].shape)
print(train_X[1])
print(train_X[1].shape)

label_Y = [np.array([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.5, 0.4, 0.02],[0.02, 0.01, 0.00, 0.00, 0.03, 0.01, 0.01, 0.01, 0.3, 0.6, 0.01]]), np.array([[-5.6],[-3.4]])] 
print(label_Y[0].shape)
print(label_Y[0])
#print(label_Y[1].shape)
#print(label_Y[1])
#print('The converted training sample is: ' + str(single_training_sample))
#print('The length of the training sample is: ' + str(len(single_training_sample)))
sample_nnetwrapper.train(train_X,label_Y)


