#suite of functions which calls different Algorithms, OMP, ROMP, l1, etc... which are not the neural networks contained in
#NNetWrapper
import numpy as np
from sklearn.linear_model import orthogonal_mp 
from cvxopt import matrix, solvers

class CSAlgorithms():
    def OMP(self, A, y, s):
    #INPUT(A,y,s) where A is the col. normalized sensing_matrix, y is observed vector, s is sparsity level. Make sure A is column normalized first.
    #OUTPUT: A set of s vectors, each of the form [(x_s, lambda), pi], where pi is a vector of zeros with a single one in the next column we will take. 
    #x_S is the solution to min_z||A_Sz-y||^2_2, and lambda is the vector A^T(A_S*x_S - y). Outputted as features, labels, where each element of features is of the form (x_s, lambda), and each member of labels is of the form pi and v, where v is propagated to every (x_s, lambda)
    #and is constructed from the final solution
    #We will be generating y and then using OMP to generate all of our training for OMPbootstrapping neural net.
    #Call from sklearn.linear_model import orthogonal_mp and use x_star = orthogonal_mp(A, y, n_nonzero_coefs = s, return_path = True)
        #Initialize empty output lists
        features = []
        labels = []
        
        #x_star returns a shape (n, s) array where each column represents the solution to min_z||A_S*z - y||^2_2
        #final_v is propagated to every label of every state
        x_star = orthogonal_mp(A, y, n_nonzero_coefs = s, return_path = True)
        if s == 1:
            x_star = np.reshape(x_star, (x_star.size, 1)) #1-sparse solutions need to be reshaped for below to work. 
        final_res = np.matmul(A,x_star[:,s-1])-y
        norm_sq = np.linalg.norm(final_res)**2
        final_v = -np.count_nonzero(x_star[:,s-1]) - 100*norm_sq
        final_v = np.reshape(final_v,(1,))

        
        #build each [(x_s, lambda), pi] sample and put them all into a list. First build [(x_s, lambda), pi] for no columns chosen case.
        x_init = np.zeros(A.shape[1]) #initialize the first sparse vector, which is a vector of zeros
        residual_init = y #initialize the first residual vector
        lambda_vec_init = np.abs(np.matmul(A.T, residual_init))#initialize the first lambda vector
        feature = [x_init, lambda_vec_init] #Compute feature for initial state of no chosen columns
        next_index = np.argmax(lambda_vec_init)
        pi = np.zeros(A.shape[1]+1) #Compute the label pi(one hot vector) of the next column chosen. +1 is for the stopping action, which is to remain consistent with alphazero algorithm. 
        pi[next_index] = 1
        label = [pi, final_v]#NOTE THAT v is the reward received AFTER we follow pi!!!
        #Add initial state feature and labels into output lists
        features.append(feature)
        labels.append(label)
        
        for i in range(x_star.shape[1]):
            #Compute lambda
            residual = y - np.matmul(A,x_star[:,i])
            lambda_vec = np.abs(np.matmul(A.T, residual))
            feature = [x_star[:,i], lambda_vec] #a length two list of np arrays
            #Compute next chosen column and extend it to a one hot vector of size n. Also compute v.
            next_index = np.argmax(lambda_vec)
            pi = np.zeros(A.shape[1]+1) #again 1 is for the stopping action. 
            pi[next_index] = 1
            label =[pi, final_v] #pi and v are both arrays here
            #add both feature and label into features and labels list
            features.append(feature)
            labels.append(label)
        return features, labels
        
    def l1(self, A, y):
        solvers.options['show_progress'] = False
        m = A[:,0].size
        n = A[0,:].size
        #A = matrix(A, (m,n), 'd')
        #Construct constraint matrix for cvx. The constraint matrix used by cvx is a list of lists, where each
        #element is a list of a column of the constraint matrix
        id_n = np.identity(n)
        zero_array1 = np.zeros((m,n))
        zero_array2 = np.zeros((n,n))
        zero_vector = np.zeros((3*n,1))
        
        negF_0 = np.hstack((-A,zero_array1))
        posF_0 = np.hstack((A,zero_array1))
        Id_negId = np.hstack((id_n, -id_n))
        negId_negId = np.hstack((-id_n, -id_n))
        zero_negId = np.hstack((zero_array2,-id_n))
        
        final_sensing_matrix = np.vstack((negF_0, posF_0, Id_negId, negId_negId, zero_negId))
        #Convert final_sensing_matrix to a list of lists, where each element in the list is a list
        #of a column of the matrix
        final_sensing_matrix = matrix(final_sensing_matrix)
        t_coeff = np.ones(n)
        t_coeff = np.reshape(t_coeff, (n,1))
        z_coeff = np.zeros(n)
        z_coeff = np.reshape(z_coeff, (n,1))
        c = np.vstack((z_coeff,t_coeff))
        c = matrix(c)
        
        y = np.reshape(y, (y.size, 1))
        final_b = np.vstack((-y, y, zero_vector))
        final_b = matrix(final_b)
        LP_solution = solvers.lp(c, final_sensing_matrix, final_b)
        final_LP_solution = np.asarray(LP_solution['x'])
        recovered_solution = final_LP_solution[0:n]
        recovered_solution = recovered_solution.flatten()
        
        return recovered_solution
        
    def rOMP(self):
        pass
        
    
        
    