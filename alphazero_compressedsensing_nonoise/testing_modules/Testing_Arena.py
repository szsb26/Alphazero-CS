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
#Initialize a new Game_args object and initialize arena
arena_game_args = Game_args()
Test_Arena = Arena(Test.coach.pnet, Test.coach.nnet, Test.game, Test.args, arena_game_args)
#Test Arena.playGame(). Play a single game. Whoever gets the higher reward is the winner. 
#Test_Arena.playGame()
#Test Arena.playGames(). Play multiple games. Returns number of wins by player 1, player 2, and number of draws. 
oneWon, twoWon, draws=Test_Arena.playGames(verbose = True)