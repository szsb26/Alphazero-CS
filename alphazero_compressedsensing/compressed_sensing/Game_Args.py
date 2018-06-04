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
    
    def generateNewObsVec(self, entry_type, sparsity): #Generate a vector x with number of nonzeros up to sparsity, with designated entry type. 
        #If self.sensing_matrix has not already been set, then this comparison throws an error. 
            
        x = np.zeros(self.sensing_matrix.shape[1])
        rand_sparsity = np.random.randint(1,sparsity)
        if entry_type == 'sdnormal':
            x[0:rand_sparsity] = np.random.normal(0, 1, rand_sparsity)
            np.random.shuffle(x)
            self.sparse_vector = x
            y = np.matmul(self.sensing_matrix,x)
            self.obs_vector = y
            
    def setIterations(self, iterations):
        self.game_iter = iterations
        
    def save_Matrix(self, filepath): #save current sensing_matrix to designated filepath
        np.save(filepath + '/sensing_matrix.npy', self.sensing_matrix)

        
