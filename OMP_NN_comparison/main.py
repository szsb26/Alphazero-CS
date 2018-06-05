#Run experiments from here with the given arguments
import os
from Test import Test

args = {#Matrix parameters
        'm': 7, #Note that sparsity is not included here because we will draw a graph where sparsity is on x axis. 
        'n':15,
        'matrices_generated': 2, #generate these number of matrices for training. These are saved and are also used to generate test signals later on.
        'max_sparsity': 7, #maximum sparsity of randomly generated sparse signals used for training and testing
        'matrix_type': 'sdnormal',
        'x_type': 'sdnormal',
        'matrix_folder': os.getcwd() + '/CS_Data' + '/sensing_matrices',
        #Select which Algorithms to compare
        'l0': False, #l0 minimization
        'OMP': True,
        'l1': True,
        'ROMP': False,
        #Neural Network Training Parameters
        'x_generated': 250000, #signals generated for each matrix generated, where each signal has max sparsity and type given above. These signals are also converted into inputs different NN's below can recognize. 
        'naive_NN': True,
            'naiveNN_hidden_neurons': 200, #10(m+n) 
            'naiveNN_lr': 0.001,
            'naiveNN_training_filepath': os.getcwd() + '/training_data' + '/naiveNN', 
            'naiveNN_generator_batchsize': 25000, #number of training samples to be used in a single call of fit. 
            'naiveNN_epochs': 50, #number of epochs in a single call of fit(number of passes to the ENTIRE dataset, 1 pass = 1 forward and 1 backwards pass)
            'naiveNN_steps_per_epoch':10, #usually is ciel(x_generated/naiveNN_generator_batchsize), equivalent to number of batches and is also equivalent to the number of yields returned in data generator. 
            'naiveNN_savedmodels_filepath': os.getcwd() + '/naivenet_model',
        'OMPbootstrap_NN': False,
            'feature_dic': {'x_l2': True, 'col_res_IP': True}, #features used for OMPbootstrapNN
            'OMPbootstrap_neurons_per_layer': 500,
            'OMPbootstrap_num_layers':2,
            'OMPbootstrap_lr':0.001,
            'OMPbootstrap_training_filepath': os.getcwd() + '/training_data' + '/OMPbootstrapNN', 
            'OMPbootstrap_generator_batchsize': 50000,
            'OMPbootstrap_epochs': 10,
            'OMPbootstrap_steps_per_epoch':20, #usually is x_generated/OMPbootstrap_generator_batchsize
            'OMPbootstrap_savedmodels_filepath': os.getcwd() + '/OMPbootstrapnet_model', 
        #Overarching Test Parameters
        'test_signals_generated': 100, # number of signals generated FOR EACH SPARSITY less than max_sparsity above, with entries type specified above which are used to test all algorithms set to True above. 
        'testData_folder': os.getcwd() + '/testData',
        }


#Load the test environment
Test = Test(args)
#Test.Data.gen_CSData(args) #1)generate matrices and their corresponding raw y, x pairs.
#Test.Data.gen_naiveNNTraining(args) #2)using the above generated data, generate data which can be directly fed into naive net
Test.Data.gen_OMPbootstrapNNTraining(args) #3)using the above generated ATA, generate data which can be directly fed into bootstrap net
#Test.trainAll() #4)For each neural network model, and for each generated matrix and their constructed training samples from steps 2) and 3), train a neural network model and save the model and the weights.



