from collections import deque
from Arena import Arena
from MCTS import MCTS
from batch_MCTS import batch_MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
from compressed_sensing.Game_Args import Game_args
import multiprocessing


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.  Game_args specified in Game_Args.py
    """
    def __init__(self, game, nnet, args, game_args):
        self.args = args
        self.game_args = game_args #Game_args object contains information about matrix A and observed vector y. THIS SINGLE GAME ARGS IS USED FOR GENERATING THE A AND y FOR SELF PLAY GAMES!!! ONLY. 
        self.arena_game_args = Game_args() #object used to generate instances of A and y across all arenas objects in learn
        self.game = game
        self.nnet = nnet
        # the competitor network. SZ: Our competitor network is just another network which plays the same game as another network 
        # and we compare which network picks the sparsest vector. The network which picks the sparsest vector is chosen and we remember these weights.
        self.pnet = self.nnet.__class__(self.args)  #past neural network. self.nnet is a NNetWrapper object, and self.pnet = self.nnet.__class__(self.args) instantiates another instance of the NNetWrapper object with self.args as input. self.pnet and self.nnet are not pointing to the same thing.                              
        #self.skip_nnet = skip_nnet
        self.batch_mcts = batch_MCTS(args, nnet) #batch_MCTS is saved here so we can call its batch prediction functions further down
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()
        
    def advanceEpisodes(self, MCTS_States_list):
    #advanceEpisodes is called in playAllGames method below.
    #plays a single move for every game/episode in MCTS_States_list
    #Takes in a list of episodes and their corresponding MCTS trees. Run exactly numMCTSsims on each episode
    #and make exactly one move for each episode. If an episode reaches a terminal node, pop it
    #from MCTS_States_list and insert it into trainExamples. Return trainExamples. trainExamples could be empty
        trainExamples = []
        new_MCTS_States_list = []
        #parallel.getActionProbs runs 'numMCTSSims' parallel searches through every MCTS tree in MCTS_States_list and returns a list of
        #action probs (which are computed from the number of visits in connected edges)
        
        actionProbs = self.batch_mcts.getActionProbs(MCTS_States_list)
        
        #using the returned actionProbs, make the next move for every episode in MCTS_States_list. IOW, for every pair (MCTS_object, States_list) in
        #MCTS_States_list, we append a new State to States_list.
        for actionProb, (temp_MCTS, temp_statesList) in zip(actionProbs, MCTS_States_list):
            
            #store the probability label and features, which were computed from the MCTS
            temp_statesList[-1].pi_as = actionProb
            
            if len(temp_statesList)-1 < self.args['tempThreshold']: 
                action = np.random.choice(len(actionProb), p = actionProb)
            else:
                action = np.argmax(actionProb)
            
            #Get the next state
            next_s = temp_MCTS.game.getNextState(temp_statesList[-1], action, temp_MCTS.game_args, 1)
            
            #Note that because we never applied traversetoLeaf on next_s, we need to call game.keyRepresentation
            nexts_keyRep = temp_MCTS.game.keyRepresentation(next_s)
            
            #Insert next_s into temp_statesList 
            temp_statesList.append(next_s)
            
            #Test if next_s is a terminal node. If next_s is a terminal node with nonzero reward, 
            #we add temp_statesList to trainExamples. Note that temp_MCTS.Es[nexts_keyRep] is well defined,
            #because if getNextState above gets us to next_s, then actionProb must have non-negative
            #probability corresponding to state next_s. This means parallel search has reached next_s before,
            #and hence temp_MCTS.Es[nexts_keyRep] is well defined. 
            r = temp_MCTS.Es[nexts_keyRep]
            
            if r != 0:
                #If r is not zero, then next_s is terminal state, so we prepare all states in temp_statesList
                #to have feature_dic, r, defined so they are ready to be fed into Neural Network. 
                #make sure every state in temp_statesList has feature_dic(feature inputs), r, and pi_as(labels) defined.
                for state in temp_statesList:
                    #set the label to the terminal reward. Note that pi_as label should all be defined already due to above.
                    #pi_as label is only not defined for the terminal state
                    state.z = r
                    #set the feature dic, x_S and Atres. 
                    state_key = self.game.keyRepresentation(state)
                    #Note that terminal states do not have their features_s dictionaries well defined since terminal states
                    #do not have to be input into the NN during search!! This is because a node
                    #is checked to be terminal or not before inputtng into NN.
                    # Hence, the else should only be executed for terminal nodes.
                    
                    if state_key in temp_MCTS.features_s:
                        state.feature_dic = temp_MCTS.features_s[state_key]
                    else:
                        state.compute_x_S(self.args, temp_MCTS.game_args)
                
                trainExamples += temp_statesList
            
            #If r == 0, then we append temp_MCTS and temp_statesList into new_MCTS_States_list
            #because the "game" is not finished yet and search continue on this pair. 
            else:
                new_MCTS_States_list.append([temp_MCTS, temp_statesList])
        
        MCTS_States_list = new_MCTS_States_list
        
        #return the new list of MCTS_States_list, with all finished episodes removed, and returns a list trainExamples,
        #which contains the states corresponding to finished episodes, with labels p_as, z computed, as well as features
        return(MCTS_States_list, trainExamples)
        
    def Initialize(self, eps):
        #Initializes MCTS_States_list
        #Initialize MCTS_States_list. Number of pairs in MCTS_States_list should equal eps_per_batch
        #used in Coach.playAllGames
        
        MCTS_States_list = []
        
        for ep in range(eps):
            #Initialize Game_args() for MCTS
            temp_game_args = Game_args()
            if self.args['fixed_matrix'] == False:
                temp_game_args.generateSensingMatrix(self.args['m'], self.args['n'], self.args['matrix_type'])
            else:
                temp_game_args.sensing_matrix = self.game_args.sensing_matrix
            
            temp_game_args.generateNewObsVec(self.args['x_type'], self.args['sparsity'])
            
            #Initialize MCTS and the first state for each MCTS
            temp_MCTS = MCTS(self.game, self.nnet, self.args, temp_game_args) 
            temp_init_state = self.game.getInitBoard(self.args, temp_game_args)
            
            #Append to MCTS_States_list
            MCTS_States_list.append([temp_MCTS, [temp_init_state]])
            
        return MCTS_States_list    
        
    def playAllGames(self, eps):
    #----------------------------------------------------------
    #While MCTS_States_list is nonempty, advance each episode in MCTS_States_list by one move.
    #continue advancing by one move until MCTS_States_list is empty, meaning that all games are completed.
    #When a game is completed, its corresponding pair should be removed from MCTS_States_list        
    #----------------------------------------------------------
        
        batchTrainExamples = []
        
        MCTS_States_list = self.Initialize(eps)
        
        while MCTS_States_list:
                        
            MCTS_States_list, trainExamples = self.advanceEpisodes(MCTS_States_list)
            #save the States_list states whose last arrived node is a terminal node. These will be used as new training samples.
            batchTrainExamples += trainExamples
            
        return batchTrainExamples   
        
    def learn(self):
        #generate or load a matrix if fixed matrix set to True. We save a Game_args object in Coach in case A is fixed so when we
        #initialize multiple MCTS objects below, we do not have to store multiple copies of A. 
                    
        if self.args['fixed_matrix'] == True:
            if self.args['load_existing_matrix'] == True:
                self.game_args.sensing_matrix = np.load(self.args['fixed_matrix_filepath'] + '/sensing_matrix.npy')
            else:
                self.game_args.generateSensingMatrix(self.args['m'], self.args['n'], self.args['matrix_type'])
                self.game_args.save_Matrix(self.args['fixed_matrix_filepath'])
        
        #keep track of learning time
        learning_start = time.time()
        
        #start training iterations
        for i in range(1, self.args['numIters']+1):
            print('------ITER ' + str(i) + '------')
            #If we are not loading a set of training data.... then:
            if not self.skipFirstSelfPlay or i>1:
                #1)Initialize empty deque for storing training data after every eps in the iteration has been processed
                iterationTrainExamples = deque([], maxlen=self.args['maxlenOfQueue'])
                
                #3)Start search. A single search consists of a synchronous search over ALL eps in the current batch.
                #Essentially the number of MCTS trees that must be maintained at once is equal to number of eps in current batch
                for j in range(self.args['num_batches']):
                    
                    iterationTrainExamples += self.playAllGames(self.args['eps_per_batch'])
            
                #Add the training samples generated in a single training iteration to self.trainExamplesHistory
                #This step is the last line included in "if not self.skipFirstSelfPlay or i>1:" block
                self.trainExamplesHistory.append(iterationTrainExamples)
            
            #Jump to here if self.skipFirstSelfPlay returns True or i<=1
            #Once iterationTrainExamples has been completed, we will use these iterationTrainExamples to retrain the Neural Network. 
            if len(self.trainExamplesHistory) > self.args['numItersForTrainExamplesHistory']:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            
            #save trainExamplesHistory 
            self.saveTrainExamples(i-1)
            
            #move all training samples from trainExamplesHistory to trainExamples for shuffling
            #shuffle trainExamples
            trainExamples = []
            for e in self.trainExamplesHistory: 
                trainExamples.extend(e)
            shuffle(trainExamples)
            
            #The Arena--------------------------------------------------------
            if self.args['Arena'] == True:
                self.nnet.save_checkpoint(folder=self.args['network_checkpoint'], filename='temp') #copy old neural network into new one
                self.pnet.load_checkpoint(folder=self.args['network_checkpoint'], filename='temp')
            
                #convert trainExamples into a format recognizable by Neural Network and train
                trainExamples = self.nnet.constructTraining(trainExamples)
                self.nnet.train(trainExamples[0], trainExamples[1])#Train the new neural network self.nnet. The weights are now updated
            
                #Pit the two neural networks self.pnet and self.nnet in the arena            
                print('PITTING AGAINST PREVIOUS VERSION')
            
                arena = Arena(self.pnet, self.nnet, self.game, self.args, self.arena_game_args) #note that Arena will pit pnet with nnet, and Game_args A and y will change constantly. Note that next iteration, arena is a reference to a different object, so old object is deleted when there are no other references to it. 
                pwins, nwins, draws = arena.playGames()
            
                print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
                if pwins+nwins > 0 and float(nwins)/(pwins+nwins) < self.args['updateThreshold']:
                    print('REJECTING NEW MODEL')
                    self.nnet.load_checkpoint(folder=self.args['network_checkpoint'], filename='temp')
                else:#saves the weights(.h5) and model(.json) twice. Creates nnet_checkpoint(i-1)_model.json and nnet_checkpoint(i-1)_weights.h5, and rewrites best_model.json and best_weights.h5
                    print('ACCEPTING NEW MODEL')
                    self.nnet.save_checkpoint(folder=self.args['network_checkpoint'], filename='nnet_checkpoint' + str(i-1))
                    self.nnet.save_checkpoint(folder=self.args['network_checkpoint'], filename='best')
            #-----------------------------------------------------------------
            
            else: #If we do not activate Arena, then all we do is just train the network, rewrite best, and write a new file 'nnet_checkpoint' + str(i-1).  
                print('TRAINING NEW NEURAL NETWORK...')
                trainExamples = self.nnet.constructTraining(trainExamples)
                    
                self.nnet.train(trainExamples[0], trainExamples[1], folder = self.args['network_checkpoint'], filename = 'trainHistDict' + str(i-1))
                      
                self.nnet.save_checkpoint(folder = self.args['network_checkpoint'], filename='nnet_checkpoint' + str(i-1))
                self.nnet.save_checkpoint(folder = self.args['network_checkpoint'], filename = 'best')
        
        #Compute total time to run alphazero
        learning_end = time.time()
        print('----------TRAINING COMPLETE----------')
        print('Total training time: ', learning_end - learning_start)

                
    def getCheckpointFile(self, iteration): #return a string which gives information about current checkpoint iteration
    #and file type (.tar)
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration): #save training examples (self.trainExamplesHistory to 
        #args['checkpoint'] folder with name of self.getCheckpointFile with given iteration. 
        folder = self.args['checkpoint']
        if not os.path.exists(folder): #if folder specified by args['checkpoint'] does not exist, then make it. 
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self): #Not used above in learn(), but will be of use in main.py
    #load training examples from folder args['load_folder_(folder)'], with filename args['load_folder_(filename)'] + '.examples'
    #OUTPUT: set self.trainExamplesHistory as the training samples in the above file, and set skipFirstSelfPlay = True
        modelFile = os.path.join(self.args['load_folder_(folder)'], self.args['load_folder_(filename)'])
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True   