#Unified Alphazero Testing wrapper module
#Not used when training alphazero and only used for testing Alphazero algorithm.
import os
import sys
sys.path.insert(0, os.getcwd() + '/compressed_sensing')
sys.path.insert(0, os.getcwd() + '/compressed_sensing/keras_tf')
from copy import deepcopy
from Coach import Coach
from Game_Args import Game_args
from NNet import NNetWrapper
from MCTS import MCTS
import numpy as np

class Test():
    def __init__(self, Alphazero):
        self.alphazero = deepcopy(Alphazero) #copy entire Alphazero object(which is a Coach object)
    
    def load_nn_model(self, model_filepath, model_filename):
        self.alphazero.nnet.load_checkpoint(model_filepath, model_filename)
    
    def predictx(self, y, s, A = None, x = None, error = 1e-3, MCTSsims = None):
        old_tempThreshold = alphazero.args['tempThreshold']
        alphazero.args['tempThreshold'] = 0
        #Change game_args
        if MCTSsims != None:
            self.alphazero.args['numMCTSSims'] = MCTSsims
        if A != None:
            self.alphazero.game_args.sensing_matrix = A
        if self.alphazero.game_args.sensing_matrix == None:
            return('Exception: No sensing_matrix has been specified in game_args')
        self.alphazero.game_args.obs_vector = y
        self.alphazero.game_args.game_iter = s
        #Initialize new MCTS tree for prediction
        self.alphazero.mcts = MCTS(alphazero.game, alphazero.nnet, alphazero.args, alphazero.game_args)
        #predict x by playing a single self play game
        trainExamples = self.alphazero.executeEpisode()
        #Compute final predicted signal by looking at the last state
        last_state = trainExamples[-1]
        predicted_x = last_state.feature_dic['x_l2']
        #if sparse vector was part of the input, then compute squared l2 error
        if x != None:
            error = np.linalg.norm(x - predicted_x)**2
        
        return predicted_x, error