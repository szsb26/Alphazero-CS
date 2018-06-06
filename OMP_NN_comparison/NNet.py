#Neural Network Wrapper for experiments. Use this to call different Neural Network structures.
from NetArch import NetArch
import csv
import numpy as np


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
        model_json = self.naive_net.model.to_json()
        with open(save_filepath + '/naiveNN_model.json', 'w') as json_file:
            json_file.write(model_json)
    #OMPbootstrapNN methods--------------------------------------------------------------------------------------------       
    def OMPbootstrapNN_DataGenerator(self, OMPbootstrap_training_filepath):
    #The generator object, used for fit.generator() in OMPbootstrapNN_train below. Refer to generator objects in python.
    #INPUT:The OMPbootstrapNN_DataGenerator generator object reads in features_y,csv and labels_x.csv from /training_data/OMPbootstrapNN
    #OUTPUT: yield returns a pair of training examples in the form of (s_t, p_as), where p_as is a zero one vector since p_as is deterministic in OMP, and s_t is a feature matrix.
    #where the first row of s_t is the solution to min_z||A_Sz - y|| and the second row of s_t is the col_res_IP A^T*(A_Sz-y)
        pass
        with open(OMPbootstrap_training_filepath + '/features' + '/NNfeature_lambda.csv', 'r') as lambda_features, open(OMPbootstrap_training_filepath + '/features' + '/NNfeature_xS.csv', 'r') as xS_features, open(OMPbootstrap_training_filepath + '/NNlabels.csv', 'r') as labels:
            while True: #loop over NNfeatures and NNlabels indefinitely since this method is a generator function
                reader_xS_features = csv.reader(xS_features)
                reader_lambda_features = csv.reader(lambda_features)
                reader_labels = csv.reader(labels)
                #Initialize what we will be outputting in yield
                Y_xS = [] #a list of 1D arrays of size n
                Y_lambda = [] #a list of 1D arrays of size n
                X = [] #a list of 1D arrays of size n (one hot vectors)
                for xS_vec, lambda_vec, pi_vec in zip(reader_xS_features, reader_lambda_features, reader_labels):
                    xS_vec = list(map(float, xS_vec))
                    lambda_vec = list(map(float, lambda_vec))
                    pi_vec = list(map(float, lambda_vec))
                    #add to dataset
                    Y_xS.append(xS_vec)
                    Y_lambda.append(lambda_vec)
                    X.append(pi_vec)
                    if len(Y_xS) >= self.args['OMPbootstrap_generator_batchsize']:
                        Y_xS = np.asarray(Y_xS)
                        Y_lambda = np.asarray(Y_lambda)
                        Y = [Y_xS, Y_lambda]
                        X = np.asarray(X)
                        yield (Y, X) #generator must yield in the form of (inputs, targets)
                        Y_xS = []
                        Y_lambda = []
                        X = []
                    #Note that at the end, some samples are thrown away...
                xS_features.seek(0)
                lambda_features.seek(0)
                labels.seek(0)
        
    def OMPbootstrapNN_train(self, train_filepath, save_filepath):  #train the naive network with the given data in training_data/OMPbootstrapNN folder.
        self.OMPbootstrap_net.fit_generator(self.OMPbootstrapNN_DataGenerator(train_filepath), steps_per_epoch = self.args['OMPbootstrap_steps_per_epoch'], epochs = self.args['OMPbootstrap_epochs'])
        self.OMPbootstrap_net.save_weights(save_filepath + 'OMPbootstrapNN_weights.h5')
        model_json = self.OMPbootstrap_net.model.to_json()
        with open(save_filepath + '/OMPbootstrapNN_model.json', 'w') as json_file:
            json_file.write(model_json)
    
    