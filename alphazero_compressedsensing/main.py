from Coach import Coach
from compressed_sensing.CSGame import CSGame
from compressed_sensing.keras.NNet import NNetWrapper

args = {
	#NN Parameters
    'lr': 0.001,
    'num_layers': 2,
    'neurons_per_layer':100,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 512,
    'num_features' : 2,
    #MCTS parameters
    'cpuct': 1, 
    'numMCTSSims': 25
    'tempThreshold': 15
}