import numpy as np
    
class Game_args(): #We save the args in Game_args class here because later on we may need class methods to generate new matrices.
    def __init__(self, A = None, y = None, x = None, k = None):
        self.sensing_matrix = A
        self.obs_vector = y
        self.sparse_vector = x #be careful that alphazero does not use this variable for prediction!! IOW, this should not appear in any other code of alphazero!!!!
        self.game_iter = k
    
    def generateSensingMatrix(self, m, n, type): #Generate sensing matrix of dim m by n of given type
        if type == 'sdnormal':
            self.sensing_matrix = np.random.randn(m,n)
            #column normalize the matrix:
            for i in range(n):
                self.sensing_matrix[:,i] = self.sensing_matrix[:,i]/np.linalg.norm(self.sensing_matrix[:,i])
        if type == 'uniform01':
            self.sensing_matrix = np.random.rand(m,n)
            for i in range(n):
                self.sensing_matrix[:,i] = self.sensing_matrix[:,i]/np.linalg.norm(self.sensing_matrix[:,i])
        if type == 'bernoulli':
            self.sensing_matrix = np.random.binomial(1,1/2,(m,n))
            self.sensing_matrix = self.sensing_matrix.astype(float)
            for i in range(n):
                self.sensing_matrix[:,i] = self.sensing_matrix[:,i]/np.linalg.norm(self.sensing_matrix[:,i])
    
    def generateNewObsVec(self, entry_type, sparsity): #Generate a vector x with number of nonzeros up to sparsity, with designated entry type. 
        #If self.sensing_matrix has not already been set, then this comparison throws an error. 
            
        x = np.zeros(self.sensing_matrix.shape[1])
        #rand_sparsity = np.random.randint(1,sparsity)
        
        rand_sparsity = np.random.randint(1,sparsity)
        
        #FOR TESTING----------------------------------
        #rand_sparsity = 650
        #print('Sparsity of generated vector is: ' + str(rand_sparsity))
        #---------------------------------------------
        
        self.game_iter = rand_sparsity
        if entry_type == 'sdnormal':
            x[0:rand_sparsity] = np.random.normal(0, 1, rand_sparsity)
        if entry_type == 'uniform01':
            x[0:rand_sparsity] = np.random.uniform(0,1,rand_sparsity)
        if entry_type == 'uniform':
            x[0:rand_sparsity] = np.random.uniform(-1,1,rand_sparsity)
            
        np.random.shuffle(x)
        self.sparse_vector = x
        y = np.matmul(self.sensing_matrix,x)
        self.obs_vector = y
            
    def setIterations(self, iterations):
        self.game_iter = iterations
        
    def save_Matrix(self, filepath): #save current sensing_matrix to designated filepath
        np.save(filepath + '/sensing_matrix.npy', self.sensing_matrix)

        