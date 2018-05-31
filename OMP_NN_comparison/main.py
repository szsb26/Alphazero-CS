#Run experiments from here with the given arguments
import os

args = {#NN Training Parameters
        'matrices_generated': 5, #generate these number of matrices for training. These matrices MUST BE SAVED FOR USE LATER IN TESTING!!!
        'max_sparsity': 5, #For each matrix generated, max_sparsity determines up to what sparsity are the graphs in Test calculated up to. 
        'x_generated': 200000, #Number of x generated for each matrix generated and for each sparsity. This determines how many training samples are generated for naive_NN and OMPbootstrap_NN. 
        'matrix_type': 'sdnormal',
        'xtype_': 'sdnormal',
        'm': 5, #Note that sparsity is not included here because we will draw a graph where sparsity is on x axis. 
        'n':15,
        'matrix_folder': os.getcwd() + '/sensing_matrices',
        #Select which Algorithms to compare
        'l0': False, #l0 minimization
        'OMP': True,
        'l1': True,
        'ROMP': False,
        'naive_NN': True,
            'naiveNN_hidden_neurons': 100, 
            'naiveNN_lr': 0.001, 
        'OMPbootstrap_NN': False,
            'feature_dic': {'x_l2': True, 'col_res_IP': True}, #features used for OMPbootstrapNN
            'bootstrap_neurons_per_layer': 50,
            'bootstrap_num_layers':2,
            'bootstrap_lr':0.001,
        #Test Class parameters(Matrices must be same as those generated above, but sparse signals can be different
        'test_signals': 500000,
        'test_max_sparsity': 5, 
        }