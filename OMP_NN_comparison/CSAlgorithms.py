#suite of functions which calls different Algorithms, OMP, ROMP, l1, etc... which are not the neural networks contained in
#NNetWrapper
import numpy as np
from sklearn.linear_model import orthogonal_mp 

class CSAlgorithms():
    def OMP(self, A, y, s):
    #INPUT(A,y,s) where A is the sensing_matrix, y is observed vector, s is sparsity level. Make sure A is column normalized first.
    #OUTPUT: A set of s vectors, each of the form [(x_s, lambda), pi], where pi is a vector of zeros with a single one in the next column we will take. 
    #x_S is the solution to min_z||A_Sz-y||^2_2, and lambda is the vector A^T(A_S*x_S - y). Outputted as features, labels, where each element of features is of the form (x_s, lambda), and each member of labels is of the form pi.
    #We will be generating y and then using OMP to generate all of our training for OMPbootstrapping neural net.
    #Call from sklearn.linear_model import orthogonal_mp and use x_star = orthogonal_mp(A, y, n_nonzero_coefs = s, return_path = True)
        #Initialize empty output lists
        features = []
        labels = []
        #build each [(x_s, lambda), pi] sample and put them all into a list. First build [(x_s, lambda), pi] for no columns chosen case.
        x_init = np.zeros(A.shape[1])
        residual_init = y
        lambda_vec_init = np.abs(np.matmul(A.T, residual_init))
        feature = [x_init, lambda_vec_init] #Compute feature for initial state of no chosen columns
        next_index = np.argmax(lambda_vec_init)
        pi = np.zeros(A.shape[1]) #Compute the label pi(one hot vector) of the next column chosen
        pi[next_index] = 1
        label = pi 
        #Add initial state feature and labels into output lists
        features.append(feature)
        labels.append(label)
        #returns a shape (n, s) array where each column represents the solution to min_z||A_S*z - y||^2_2
        x_star = orthogonal_mp(A, y, n_nonzero_coefs = s, return_path = True)
        if s == 1:
            x_star = np.reshape(x_star, (x_star.size, 1)) #1-sparse solutions need to be reshaped for below to work. 
        for i in range(x_star.shape[1]):
            #Compute lambda
            residual = y - np.matmul(A,x_star[:,i])
            lambda_vec = np.abs(np.matmul(A.T, residual))
            feature = [x_star[:,i], lambda_vec] 
            #Compute next chosen column and extend it to a one hot vector of size n. 
            next_index = np.argmax(lambda_vec)
            pi = np.zeros(A.shape[1])
            pi[next_index] = 1
            label = pi
            #add both feature and label into features and labels list
            features.append(feature)
            labels.append(label)
        return features, labels
        
    def l1(self):
        pass
        
    def rOMP(self):
        pass
        
    
        
    