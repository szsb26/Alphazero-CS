#Neural Network Wrapper for experiments. Use this to call different Neural Network structures.
from NetArch import NetArch
import csv
import numpy as np
from keras.models import model_from_json


class NNetWrapper():
    def __init__(self, args):
        self.args = args
        self.netWrapper = NetArch()
        self.naive_net = self.netWrapper.naiveNet(args) #compiled model object
        self.OMPbootstrap_net = self.netWrapper.OMPbootstrap_Net(args) #compiled model object
    #NaiveNN methods-------------------------------------------------------------------------------------------------    
    def NN_DataGenerator(self, naiveNN_training_filepath): 
    #The generator object, used for fit.generator() in naiveNN_train below. Refer to generator objects in python.
    #INPUT:The NN_DataGenerator generator object reads in features_y.csv and labels_x.csv from given filepath
    #OUTPUT: yield returns a batch of training examples in the form of (Y,X). 
        with open(naiveNN_training_filepath + '/NNfeatures_y.csv', 'r') as features, open(naiveNN_training_filepath + '/NNlabels_x.csv', 'r') as labels:
            while True: #loops indefinitely over NNfeatures_y and NNlabels_x 
                #Initialize reader_features, reader_labels, Y, X
                reader_features = csv.reader(features)
                reader_labels = csv.reader(labels)
                Y = []
                X = []
                for row_features, row_labels in zip(reader_features, reader_labels): #we can only iterate over csv.reader objects once!!!!
                    row_features = list(map(float, row_features))
                    row_labels = list(map(float, row_labels))
                    Y.append(row_features)
                    X.append(row_labels)
                    if len(Y) >= self.args['naiveNN_generator_batchsize']:
                        Y = np.asarray(Y)
                        X = np.asarray(X)
                        yield (Y, X) #yield returns (Y,X) at this point, and next time we call NN_DataGenerator(), the code continues from here!!
                        Y = []
                        X = []
                Y = np.asarray(Y)
                X = np.asarray(X)
                yield (Y,X)
                features.seek(0)#resets iterators to the beginning
                labels.seek(0)#resets iterators to the beginning of file
    
    def naiveNN_train(self, train_filepath, save_filepath): #train the naive network with the given data in training_data/naiveNN folder and save the model/weights
        self.naive_net.fit_generator(self.NN_DataGenerator(train_filepath), steps_per_epoch = self.args['naiveNN_steps_per_epoch'], epochs = self.args['naiveNN_epochs']) #steps_per_epoch is the total number of yields received from NN_DataGenerator
        self.naive_net.save_weights(save_filepath + '/naiveNN_weights.h5')
        model_json = self.naive_net.to_json()
        with open(save_filepath + '/naiveNN_model.json', 'w') as json_file:
            json_file.write(model_json)
            
    def load_naiveNN_model(self, filepath_model, filepath_weights):
    #FUNCTION: Load the model and weights into self.naive_net
        json_file = open(filepath_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.naive_net = model_from_json(loaded_model_json)
        #self.naive_net.compile(loss = 'binary_crossentropy', optimizer = adam_custom, metrics = ['accuracy'])
        self.naive_net.load_weights(filepath_weights)
        
    #OMPbootstrapNN methods--------------------------------------------------------------------------------------------       
    def OMPbootstrapNN_DataGenerator(self, OMPbootstrap_training_filepath):
    #The generator object, used for fit.generator() in OMPbootstrapNN_train below. Refer to generator objects in python.
    #INPUT:The OMPbootstrapNN_DataGenerator generator object reads in features_y,csv and labels_x.csv from /training_data/OMPbootstrapNN
    #OUTPUT: yield returns a pair of training examples in the form of (s_t, p_as), where p_as is a zero one vector since p_as is deterministic in OMP, and s_t is a feature matrix.
    #where the first row of s_t is the solution to min_z||A_Sz - y|| and the second row of s_t is the col_res_IP A^T*(A_Sz-y)
        pass
        with open(OMPbootstrap_training_filepath + '/features' + '/NNfeature_lambda.csv', 'r') as lambda_features, open(OMPbootstrap_training_filepath + '/features' + '/NNfeature_xS.csv', 'r') as xS_features, open(OMPbootstrap_training_filepath + '/labels' + '/NNlabel_pas.csv', 'r') as pas_labels, open(OMPbootstrap_training_filepath + '/labels' + '/NNlabel_v.csv', 'r') as v_labels:
            while True: #loop over NNfeatures and NNlabels indefinitely since this method is a generator function
                reader_xS_features = csv.reader(xS_features)
                reader_lambda_features = csv.reader(lambda_features)
                reader_labels_pas = csv.reader(pas_labels)
                reader_labels_v = csv.reader(v_labels)
                #Initialize what we will be outputting in yield
                Y_xS = [] #a list of 1D arrays of size n
                Y_lambda = [] #a list of 1D arrays of size n
                X_pas = [] #a list of 1D arrays of size n (one hot vectors)
                X_v = [] #a list of scalars 
                for xS_vec, lambda_vec, pi_vec, v_scalar in zip(reader_xS_features, reader_lambda_features, reader_labels_pas, reader_labels_v):
                    xS_vec = list(map(float, xS_vec))
                    lambda_vec = list(map(float, lambda_vec))
                    pi_vec = list(map(float, pi_vec))
                    v_scalar = list(map(float, v_scalar))
                    #add to dataset
                    Y_xS.append(xS_vec)
                    Y_lambda.append(lambda_vec)
                    X_pas.append(pi_vec)
                    X_v.append(v_scalar)
                    if len(Y_xS) >= self.args['OMPbootstrap_generator_batchsize']:
                        Y_xS = np.asarray(Y_xS)
                        Y_lambda = np.asarray(Y_lambda)
                        Y = [Y_xS, Y_lambda]
                        X_pas = np.asarray(X_pas)
                        X_v = np.asarray(X_v)
                        X = [X_pas, X_v]
                        yield (Y, X) #generator must yield in the form of (inputs, targets)
                        Y_xS = []
                        Y_lambda = []
                        X_pas = []
                        X_v = []
                    #Note that at the end, some samples are thrown away...
                xS_features.seek(0)
                lambda_features.seek(0)
                pas_labels.seek(0)
                v_labels.seek(0)
        
    def OMPbootstrapNN_train(self, train_filepath, save_filepath):  #train the naive network with the given data in training_data/OMPbootstrapNN folder.
        self.OMPbootstrap_net.fit_generator(self.OMPbootstrapNN_DataGenerator(train_filepath), steps_per_epoch = self.args['OMPbootstrap_steps_per_epoch'], epochs = self.args['OMPbootstrap_epochs'])
        self.OMPbootstrap_net.save_weights(save_filepath + '/OMPbootstrapNN_weights.h5')
        model_json = self.OMPbootstrap_net.to_json()
        with open(save_filepath + '/OMPbootstrapNN_model.json', 'w') as json_file:
            json_file.write(model_json)
            
            
    def load_OMPbootstrapNN_model(self, filepath_model, filepath_weights):
        #FUNCTION: Load the model and weights into self.OMPbootstrap_net corresponding to appropriate generated matrix
        #Load the model with given filepath
        json_file = open(filepath_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.OMPbootstrap_net = model_from_json(loaded_model_json)
        #self.OMPbootstrap_net.compile(loss=['categorical_crossentropy','mean_squared_error'], metrics=['accuracy'], optimizer=Adam(args['OMPbootstrap_lr']))
        #Load the weights with given filepath
        self.OMPbootstrap_net.load_weights(filepath_weights)
    
    
    def OMPbootstrapNN_predict_x(self, A, y, s):
        #FUNCTION: given that the appropriate model and weights have been loaded, predict x given A, y, and s
        #returns the final predicted signal x. 
	    #Initialize
	    iter = 0
	    x = np.zeros(A.shape[1]) #keeps track of current x solution
	    S = [] #columns_taken
	    res = y
	    lambda_vec = np.matmul(A.T, y)
	    
	    x = np.reshape(x, (1,A.shape[1])) #reshape for prediction below
	    lambda_vec = np.reshape(lambda_vec, (1, A.shape[1]))
	    
	    while(iter < s):
	        #Output prediction of what column to take next
	        pas, v = self.OMPbootstrap_net.predict([x,lambda_vec])
	        #Compute next input state. It could be that neural network outputs an action already taken...Hence we use a set data structure for col_taken here to avoid searching whether we have already chosen a column or not
	        next_index = np.argmax(pas) #find the next index
	        S.append(next_index) #remember the list of taken columns
	        A_S = A[:,S]
	        x_star = np.linalg.lstsq(A_S, y)
	        #Compute the new x
	        i = 0
	        for k in S:
	            x[0][k] = x_star[0][i]
	            i += 1
	        x_flattened = x.flatten()
	        res = y - np.matmul(A, x_flattened) #update residual. Note that x is shape (1, A.shape[1])
	        lambda_vec = np.matmul(A.T, res) #update lambda
	        lambda_vec = np.reshape(lambda_vec, (1, A.shape[1])) #reshape lambda for feeding into neural net
	        
	        iter += 1 #update iter
	    
	    return x
    
    