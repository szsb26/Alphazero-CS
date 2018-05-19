import sys
import numpy as np
	
class Game_args(): #We save the args in Game_args class here because later on we may need class methods to generate new matrices.
	def __init__(self, A, y, m):
		self.sensing_matrix = A
		self.obs_vector = y
		self.game_iter = m
	
	def generateSensingMatrix(type): #Generate sensing matrix according to type
		
		self.sensing_matrix = None
		
	def generateNewObsVec():
		
		self.obs_vector = None
	