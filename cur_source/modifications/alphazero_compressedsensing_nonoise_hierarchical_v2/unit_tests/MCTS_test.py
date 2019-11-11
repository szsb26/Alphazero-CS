import sys
import os
import numpy as np

#import necessary libraries
sys.path.append("..")
sys.path.append("../compressed_sensing")
sys.path.append("../compressed_sensing/keras_tf")

from Arena import Arena
from MCTS import MCTS
from CSGame import CSGame
from Game_Args import Game_args
from CSState import State

from NNet import NNetWrapper


#import general libraries
from batch_MCTS import batch_MCTS


#THe advanceEpisodes function copied over from Coach
#--------------------------------------------------------------------------------------------------------------
def advanceEpisodes(MCTS_States_list, Threaded_MCTS, args):
#plays a single move for every game/episode in MCTS_States_list
#Takes in a list of episodes and their corresponding MCTS trees. Run exactly numMCTSsims on each episode
#and make exactly one move for each episode. If an episode reaches a terminal node, pop it
#from MCTS_States_list and insert it into trainExamples. Return trainExamples. trainExamples could be empty
    trainExamples = []
    #parallel.getActionProbs runs 'numMCTSSims' parallel searches through every MCTS tree in MCTS_States_list and returns a list of
    #action probs (which are computed from the number of visits in connected edges)
    
    actionProbs = Threaded_MCTS.getActionProbs(MCTS_States_list, temp = 1)
        
    #TESTING-------------------
    print('')
    print('actionProbs in advanceEpisodes has been computed correctly', actionProbs)
    print('')
    print('Preparing to move root for every MCTS object...')
    #END TESTING---------------
    
    #using the returned actionProbs, make the next move for every episode in MCTS_States_list
    for actionProb, pair in zip(actionProbs, MCTS_States_list):
        #Note that pair = (MCTS object, [list of States]), ep[1][-1] is the last state object
        temp_MCTS = pair[0]
        temp_statesList = pair[1]
        #get the hashkey of the last state in temp_statesList
        temp_statekeyRep = temp_statesList[-1].keyRep
            
        #store the probability label and features, which were computed from the MCTS
        temp_statesList[-1].pi_as = actionProb
            
        action = np.random.choice(len(actionProb), p = actionProb)
            
        #Get the next state
        next_s = temp_MCTS.game.getNextState(temp_statesList[-1], action)
            
        #if length of column indices is greater than maxTreeDepth, reassign next_s to the state in Rsa instead. Otherwise, keep next_s
        #if len(next_s.col_indices) > args['maxTreeDepth'] and next_s.action_indices[-1] == 0:
            #next_s = temp_MCTS.Rsa[(temp_statekeyRep, action)]
            
        nexts_keyRep = temp_MCTS.game.keyRepresentation(next_s)
        #Insert next_s into temp_statesList 
        
        temp_statesList.append(next_s)
            
        #Test if next_s is a terminal node. If next_s is a terminal node with nonzero reward, 
        #we add temp_statesList to trainExamples
        r = temp_MCTS.Es[nexts_keyRep]
        
        #FOR TESTING--------------------------
        print('r: ', r)
        #END TESTING--------------------------
        
        if r != 0:
            #If r is not zero, then nexts is terminal state, so we prepare all states in temp_statesList
            #to have feature_dic, r, defined so they are ready to be fed into Neural Network. 
            #make sure every state in temp_statesList has feature_dic(feature inputs), r, and pi_as(labels) defined.
            for state in temp_statesList:
                #set the label to the terminal reward. Note that pi_as label should all be defined already for nonterminal states.
                state.z = r
                #set the feature dic, x_S and Atres. 
                state_key = temp_MCTS.game.keyRepresentation(state)
                #Note that terminal states DO NOT have their feature_dic well defined!
                if state_key in temp_MCTS.features_s:
                    state.feature_dic = temp_MCTS.features_s[state_key]
                else:
                    state.compute_x_S_and_res(args, temp_MCTS.game_args)
            trainExamples += temp_statesList
                
    #create new (MCTS tree object, [list of States]) list and reassign. Gets rid of pairs (MCTS_object, [States list]) such that the root has already traversed to a terminal node. 
    new_MCTS_States_list = []
    for pair in MCTS_States_list:
        last_state = pair[1][-1]
        temp_MCTS = pair[0]
        if last_state.z == None: #This implies that the last state was not a terminal node, so reward label for every state have 
        #not been defined yet
            new_MCTS_States_list.append(pair)
        
    MCTS_States_list = new_MCTS_States_list
        
    #return the new list of MCTS_States_list, with all finished episodes removed, and returns a list trainExamples,
    #which contains the states corresponding to finished episodes, with labels p_as, z computed, as well as features
    return(MCTS_States_list, trainExamples)
#--------------------------------------------------------------------------------------------------------------

args = {
    #Compressed Sensing Parameters, Ax = y, where A is of size m by n
    'fixed_matrix': True, #fixes a single matrix across entire alphazero algorithm. If set to False, then self play games generated in each iteration have different sensing matrices. The below options will not run if this is set to False.
        'load_existing_matrix': False, #If we are using a fixed_matrix, then this option toggles whether to load an existing matrix from args['fixed_matrix_filepath'] or generate a new one. If loading an existing matrix, the matrix must be saved as name 'sensing_matrix.npy'
            'matrix_type': 'sdnormal',  #type of random matrix generated if(assuming we are not loading existing matrix)
    'x_type': 'uniform01',  #type of entries generated for sparse vector x when playing games of self-play
    'm': 7, #row dimension of A
    'n': 15, #column dimension of A
    'sparsity':7, #dictates the maximum sparsity of x when generating the random vector x. Largest number of nonzeros of x is sparsity-1. sparsity cannot be greater than m above. 
    'fixed_matrix_filepath': os.getcwd() + '/fixed_sensing_matrix', #If args['fixed_matrix'] above is set to True, then this parameter determines where the fixed sensing matrix is saved or where the existing matrix is loaded from. 
    #---------------------------------------------------------------
    #General Alphazero Training Parameters
    'num_batches': 1,
    'eps_per_batch': 2, 
    'numIters': 100, #number of alphazero iterations performed. Each iteration consists of 1)playing numEps self play games, 2) retraining neural network
    'numEps': 400, #dictates how many self play games are played each iteration of the algorithm
    'maxlenOfQueue':10000, #dictates total number of game states saved(not games). 
    'numItersForTrainExamplesHistory': 1, #controls the size of trainExamplesHistory, which is a list of different iterationTrainExamples deques. 
    'checkpoint': os.getcwd() + '/training_data', #filepath for SAVING newly generated self play training data
    'load_training': False, #If set to True, then load latest batch of self play games for training. 
        'load_folder_(folder)': os.getcwd() + '/training_data', #filepath for LOADING the latest set of training data
        'load_folder_(filename)': 'best.pth.tar', #filename for LOADING the latest generated set of training data. Currently, this must be saved as 'best.pth.tar'
    'Arena': False, #determines whether model selection/arena is activated or not. Below options will not be run if this is set to False.
        'arenaCompare': 100, #number of games played in the arena to compare 2 networks pmcts and nmcts
        'updateThreshold': 0.55, #determines the percentage of games nmcts must win for us to update pmcts to nmcts
    #---------------------------------------------------------------
    #NN Parameters
    'lr': 0.001,    #learning rate of NN, relevant for NetArch(), NetArch1()
    'num_layers': 2,    #number of hidden layers after the 1st hidden layer, only relevant for NetArch()
    'neurons_per_layer':200,    #number of neurons per hidden layer
    'epochs': 10,   #number of training epochs. If There are K self play states, then epochs is roughly K/batch_size. Note further that K <= numEps*sparsity. epochs determines the number of times weights are updated.
    'batch_size': 400, #dictates the batch_size when training 
    'num_features' : 2, #number of self-designed features used in the input
    'load_nn_model' : False, #If set to True, load the best network (best_model.json and best_weights.h5)
    'network_checkpoint' : os.getcwd() + '/network_checkpoint', #filepath for SAVING the temp neural network model/weights, checkpoint networks model/weights, and the best networks model/weights
    #features: True if we wish to use as a feature, False if we do not wish to use as a feature
    'x_l2' : True,      #solution to min_z||A_Sz - y||_2^2, where A_S is the submatrix of columns we have currently chosen
    'lambda' : True,    #the vector of residuals, lambda = A^T(A_Sx-y), where x is the optimal solution to min_z||A_Sz - y||_2^2
    #---------------------------------------------------------------
    #MCTS parameters
    'cpuct': 2, #controls the amount of exploration at each depth of MCTS tree.
    'numMCTSSims': 50, #For each move, numMCTSSims is equal to the number of MCTS simulations in finding the next move during self play. Smallest value of numMCTSsims is 2.    
    'maxTreeDepth': 30, #sets the max tree depth of MCTS search. Once max tree depth is reached, and if sparsity > maxTreeDepth, then bootstrapped network(skip_rule) is used to pick remainder of the columns.Note that this means that maxTreeDepth does not count the root or terminal nodes as levels in the tree. This means real tree depth must add 2.
    'skip_rule': None, #Current options: None(defaults to current policy/value network), OMP(uses OMP rule to pick next column), bootstrap(uses boostrapped network in bootstrap folder) 
            'skip_nnet_folder': os.getcwd() + '/skip_network', 
            'skip_nnet_filename': 'skip_nnet', 
    'beta': 1, #Recall the augmented probability aug_prob = beta * probs + (1-beta) * 1/(len(x)) * x_I, where x_I is the indicator vector of ones of the true sparse solution x. Hence, higher beta values increase the probabilities towards choosing the correct column choices. 
                 #SET beta = 1 DURING TESTING SINCE x SHOULD BE UNKNOWN DURING TESTING. 
    'tempThreshold': 25,    #dictates when the MCTS starts returning deterministic polices (vector of 0 and 1's). See Coach.py for more details.
    'gamma': 1, #note that reward for a terminal state is -alpha||x||_0 - gamma*||A_S*x-y||_2^2. The smaller gamma is, the more likely algorithm is going to choose stopping action earlier(when ||x||_0 is small). gamma enforces how much we want to enforce Ax is close to y. 
                #choice of gamma is heavily dependent on the distribution of our signal and the distribution of entries of A. gamma should be apx. bigger than m/||A_Sx^* - y||_2^2, where y = Ax, and x^* is the solution to the l2 regression problem.
    'alpha':1e-5,  #note that reward for a terminal state is -alpha||x||_0 - gamma*||A_S*x-y||_2^2. The smaller alpha is, the more weight the algorithm gives in selecting a sparse solution.
    'epsilon': 1e-5, #If x is the optimal solution to l2, and the residual of l2 regression ||A_Sx-y||_2^2 is less than epsilon, then the state corresponding to indices S is a terminal state in MCTS. 
}


#Test the search capabilities of multiple MCTS objects using Threading_MCTS

#global Game_args object, global CSGame(for game rules and such), global policy/value net
game_args = Game_args()
game_args.generateSensingMatrix(args['m'], args['n'], args['matrix_type'])

Game = CSGame()
nnet = NNetWrapper(args)


#---------------------------------------------------
#Initialize MCTS_States_list
for i in range(args['num_batches']):
    MCTS_States_list = []
    batchTrainExamples = []
    
    #In loop below, we create a pair in the form of (MCTS_object, [list of States])
    for ep in range(args['eps_per_batch']):
        #Initialize Game_args() for MCTS
        temp_game_args = Game_args()
        temp_game_args.sensing_matrix = game_args.sensing_matrix
        temp_game_args.generateNewObsVec(args['x_type'], args['sparsity'])
        #Initialize MCTS object
        temp_MCTS = MCTS(Game, nnet, args, temp_game_args, identifier = int(str(i) + str(ep)))
        #Initialize root node for the MCTS object.
        #Note that ep is an identifier telling us which MCTS tree the states belong to
        temp_init_state = Game.getInitBoard(args, temp_game_args, identifier = int(str(i) + str(ep)))
        
        #append [temp_MCTS, temp_init_state] to MCTS_States_list
        MCTS_States_list.append([temp_MCTS, [temp_init_state]])
        
        #FOR TESTING---------------------
        print('')
        print('---------INITIALIZATION----------')
        print('MCTS object identifier: ', temp_MCTS.identifier)
        print('y', temp_game_args.obs_vector)
        print('x', temp_game_args.sparse_vector)
        print('')
        
        #END TESTING---------------------
        
#Print some tests about MCTS_States_list
#---------------------------------------------------
#Initialize threading_MCTS object for advanceEpisodes in Coach
threaded_mcts = Threading_MCTS(args, nnet)

#Test advanceEpisodes
#FOR TESTING----------------------------------------
print('-----------------------FINAL STATISTICS FOR MCTS_States_list BEFORE advanceEpisodes-----------------------' )
for pair in MCTS_States_list:
    MCTS_obj = pair[0]
    State_list = pair[1]
    print('')
    print('1) Statistics for MCTS object ' + str(MCTS_obj.identifier))
    print('MCTS Nsa: ', MCTS_obj.Nsa)
    print('MCTS Qsa: ', MCTS_obj.Qsa)
    print('MCTS Ns: ', MCTS_obj.Ns)
    print('MCTS Ps: ', MCTS_obj.Ps)
    print('MCTS Vs: ', MCTS_obj.Vs)
    print('')
    print('2) Statistics for States_list')
    for State in State_list:
        print('State action indices: ', State.action_indices)
        print('State feature dic: ', State.feature_dic)
        print('State p_as label: ', State.pi_as)
        print('State.z reward label: ', State.z)
#END TESTING----------------------------------------

MCTS_States_list, trainExamples = advanceEpisodes(MCTS_States_list, threaded_mcts, Game, args)

#FOR TESTING----------------------------------------
print('-----------------------FINAL STATISTICS FOR MCTS_States_list AFTER advanceEpisodes-----------------------' )
for pair in MCTS_States_list:
    MCTS_obj = pair[0]
    State_list = pair[1]
    print('')
    print('1) Statistics for MCTS object ' + str(MCTS_obj.identifier))
    print('MCTS Nsa: ', MCTS_obj.Nsa)
    print('MCTS Qsa: ', MCTS_obj.Qsa)
    print('MCTS Ns: ', MCTS_obj.Ns)
    print('MCTS Ps: ', MCTS_obj.Ps)
    print('MCTS Vs: ', MCTS_obj.Vs)
    print('')
    print('2) Statistics for States_list')
    for State in State_list:
        print('State action indices: ', State.action_indices)
        print('State identifier: ', State.identifier)
        print('State feature dic: ', State.feature_dic)
        print('State p_as label: ', State.pi_as)
        print('State.z reward label: ', State.z)
        
print('-----------------------FINAL STATISTICS FOR trainExamples AFTER advanceEpisodes-----------------------' )
for State in trainExamples:
    print('')
    print('State action indices: ', State.action_indices)
    print('State identifier: ', State.identifier)
    print('State feature dic: ', State.feature_dic)
    print('State p_as label: ', State.pi_as)
    print('State.z reward label: ', State.z)
    print('')
#END TESTING----------------------------------------
