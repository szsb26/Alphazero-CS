import sys
from Test_Class import Test
import numpy as np
sys.path.insert(0,'/Users/sichenzhong/Desktop/Sichen/Graduate_School/ML/NN_MCTS_CS/python_src/alphazero_compressedsensing')
sys.path.insert(0,'/Users/sichenzhong/Desktop/Sichen/Graduate_School/ML/NN_MCTS_CS/python_src/alphazero_compressedsensing/compressed_sensing')

from MCTS import MCTS
from Arena import Arena
from Game_Args import Game_args

#Load the testing environment
Test = Test()
#Define pmcts and nmcts
pmcts = MCTS(Test.game, Test.coach.pnet, Test.args, Test.game_args)#pnet and nnet should be the same here(same model initialized with same weights)
nmcts = MCTS(Test.game, Test.coach.nnet, Test.args, Test.game_args) 
#Initialize a new Game_args object and initialize arena
arena_game_args = Game_args()
Test_Arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)), lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), Test.game, Test.args, arena_game_args)
#Test Arena.playGame(). Play a single game. Whoever gets the higher reward is the winner. 
#Test_Arena.playGame()
#Test Arena.playGames()
oneWon, twoWon, draws=Test_Arena.playGames(verbose = True)