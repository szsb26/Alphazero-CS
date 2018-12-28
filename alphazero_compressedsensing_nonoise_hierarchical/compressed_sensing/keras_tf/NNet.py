from CS_NNet import NetArch, NetArch1, NetArch2
import numpy as np
import pickle
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from numpy.random import seed
seed(1)

class NNetWrapper(): 
    def __init__(self, args): 
        self.args = args
        #self.nnet is a NetArch object.
        #NetArch object contains self variables self.args, self.p_as, self.v, and most importantly, self.model
        self.nnet = NetArch2(args)
    
    def constructTraining(self, states): #this method is used in Coach.py
    #INPUT: a list of state objects which have values for self.feature_dic, self.p_as, and self.z
    #OUTPUT: A list [X,Y] training data saved into .csv file. Ideally for training, we would just directly read in from .csv file if
    #necessary
    #NOTE: Instead of using features_dic, we can also stack all state's state.nn_inputs. 
        num_states = len(states)
        X = []
        Y = []
        #Initialize every entry in X as an empty numpy array matrix     
        for key in states[0].feature_dic: #Each state's feature dictionary should contain vectors which are all the same size
            zero_feature_matrix = np.empty((num_states,len(states[0].feature_dic[key])))
            X.append(zero_feature_matrix)
    
        #Fill in the rows of each empty matrix in list X
        for i in range(num_states): #iterate over the number of states, which is equal to the row dim of every np array in X.
            list_index = 0
            for key in states[i].feature_dic:
                X[list_index][i][:] = states[i].feature_dic[key] 
                list_index += 1 
                
        #Construct labels Y, which is length 2 list of numpy arrays
        pi_as_empty = np.empty((num_states, states[0].action_indices.size))
        z_empty = np.empty((num_states,1))
        Y.append(pi_as_empty)
        Y.append(z_empty)
            
        for i in range(num_states): #Y[0].shape equals number of states
            Y[0][i][:] = states[i].pi_as
            Y[1][i][0] = states[i].z
        
        converted_training = [X, Y]
        return converted_training
            
    def train(self, X, Y, folder = None, filename = None): 
    #INPUT: A list, where each element in the list is itself a list of self play games. 
    #Each element in this embedded list is in the form [X,Y], which contains all the data for a single
    #self-pair game. 
    #OUTPUT: updated class variable self.nnet, plus a pickle object of the history saved in filepath given in args.
        
        history = self.nnet.model.fit(X,Y, epochs = self.args['epochs'], batch_size = self.args['batch_size'])
        
        if folder != None and filename != None:
            with open(folder + '/' + filename, 'wb') as file_pi:
                pickle.dump(history.history, file_pi) 
        
        
    def predict(self, state): 
    #INPUT: state object. Note that state.col_indices, state.feature_dic and state.converttoNNInput 
    #must all be computed before NNetWrapper.predict can make a meaningful prediction. 
    #OUTPUT: p_as and v, where p_as is a numpy array of shape(args['n']+1, ) and v is a scalar value(float)
        
        if state.nn_input == None:
            print('nn_input of input state has not been computed')
            return
        else: 
            p_as, v = self.nnet.model.predict(state.nn_input)#p_as.shape(1,n+1), v.shape = (1,1)
            p_as = p_as.flatten() #change array shape to (n+1,). 
            v = v[0][0] #v is now a scalar instead of a matrix of size (1,1)
            
            return p_as, v
    
    def save_checkpoint(self, folder, filename):
    #INPUT: folder and filename 
    #OUTPUT: None
    #FUNCTION: save the current model and its weights in some folder
        self.nnet.model.save_weights(folder + '/' + filename + '_weights.h5')
        model_json = self.nnet.model.to_json()
        with open(folder + '/' + filename + '_model.json', 'w') as json_file:
            json_file.write(model_json)
            
    def load_checkpoint(self, folder, filename):
    #INPUT: folder and filename
    #OUTPUT: load a model and its weights with given folder and filename
        #Load the model
        json_file = open(folder + '/' + filename + '_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.nnet.model = model_from_json(loaded_model_json)
        #Load the weights
        self.nnet.model.load_weights(folder + '/' + filename + '_weights.h5')
        #Recompile the model
        self.nnet.model.compile(loss=['categorical_crossentropy','mean_squared_error'], metrics=['accuracy'], optimizer=Adam(self.args['lr']))
        
