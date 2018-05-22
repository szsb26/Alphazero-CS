import sys
import numpy as np
	
class Game_args(): #We save the args in Game_args class here because later on we may need class methods to generate new matrices.
	def __init__(self, A = None, y = None, x = None, k = None):
		self.sensing_matrix = A
		self.obs_vector = y
		self.sparse_vector = x
		self.game_iter = k
	
	def generateSensingMatrix(self, m, n, type): #Generate sensing matrix of dim m by n of given type
		if type == 'sdnormal':
			self.sensing_matrix = np.random.randn(m,n)
		if type == 'uniform01':
			self.sensing_matrix = np.random.rand(m,n)
	
	def generateNewObsVec(self):
		if self.sensing_matrix == None:
			return ('Sensing Matrix has not been set!')
			
		x = np.zeros(self.sensing_matrix.shape[1])
		if entry_type == 'sdnormal':
			x[0:sparsity] = np.random.normal(0, 1)
			np.random.shuffle(x)
			self.sparse_vector = x
			y = np.matmul(self.sensing_matrix,x)
			self.obs_vector = y
			
	def setIterations(self, iterations):
		self.game_iter = iterations
	