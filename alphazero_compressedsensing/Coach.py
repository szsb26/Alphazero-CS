from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.  Game_args specified in Game_Args.py
    """
    def __init__(self, game, nnet, args, Game_args):
        self.game = game
        self.nnet = nnet	#new neural network
        # the competitor network. SZ: Our competitor network is just another network which plays the same game as another network 
        # and we compare which network picks the sparsest vector. The network which picks the sparsest vector is chosen and we remember these weights.
        self.pnet = self.nnet.__class__(self.game)	#past neural network								
        self.args = args
        self.game_args = Game_args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        #INPUT: None
        #
        #OUTPUT: list of state objects, where for every state, state.pi_as, state.z, and state.feature_dic have all been
        #updated
        #FUNCTION:We play a single game until we reach a end/terminal state. Each state
        #is saved and the policy distribution is also saved. When we reach a terminal state,
        #we propagate the final result z to each state object. In this method, each state object 
        #has state.feature_dic, state.pi_as, and state.z updated. Finally every state in states is
        #converted into a format recognizable by the Neural Network. 
        
        state = self.game.getInitBoard() #State Object
        action_size = self.game.getActionSize()
        states = [] #will convert all states into X using NNet.convertStates
        trainExamples = []
        
		#After episodeStep number of played moves into a single game (>= tempTHreshold), MCTS.getActionProb
        #starts returning deterministic policies pi. Hence, the action chosen to get the next state
        #for our root node will be deterministic. The higher the integer tempThreshold is,
        #the more randomness introduced in generating our training samples before move tempThreshold.
        
        episodeStep = 0
        
        while True:
            episodeStep += 1
            temp = int(episodeStep < self.args['tempThreshold']) #int(True) = 1, o.w. 0
			
			#Note that MCTS.getActionProb runs a fixed number of MCTS.search determined by 
			#args.MCTSSims
            pi = self.mcts.getActionProb(state, temp=temp)
            state.pi_as = pi #update the label pi_as
            #Construct the States_List and Y
            states.append(state)
			#choose a random action (integer) with prob in pi.
            action = np.random.choice(len(pi), p=pi)
            #Given the randomly generated action, move the root node to the next state.   
            state = self.game.getNextState(state, action)

            r = self.game.getGameEnded(state, self.game_args)
			
			#return breaks out of the while loop. If r not equal to 0, that means the state we are
			#on is a terminal state, which implies we should propagate the rewards up to every 
			#state in states
            if r!=0:
                for state in states:
                	#compute state.feature_dic
                	state.compute_x_S_and_res(self.args, self.game_args)
                	#compute the label state.z
                	state.z = r
                trainExamples = states 
                return trainExamples #returns a list of state objects with features, labels all computed

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args['numIters']+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args['maxlenOfQueue'])
                #bookkeeping objects contained in pytorch_classification.utils
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args['numEps'])
                end = time.time()
                #IMPORTANT PART OF THE CODE. GENERATE NEW A AND NEW y HERE
                #-----------------------------------------------------
                for eps in range(self.args['numEps']):
                	#Initialize a new game by setting A, x, y, and then execute a single game of self play with self.executeEpisode()
                	self.game_args.generateSensingMatrix(self.args['m'], self.args['n'], self.args['matrix_type']) #generate a new sensing matrix
                	self.game_args.generateNewObsVec('x_type')	#generate a new observed vector y
                    self.mcts = MCTS(self.game, self.nnet, self.args)   #reset search tree for each game we play
                    iterationTrainExamples += self.executeEpisode() #Play a new game with newly generated y. iterationTrainExamples is a deque containing states each generated self play game
                #-----------------------------------------------------
                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                # save the iteration examples to the history 
                #self.trainExamplesHistory is a list of deques, where each deque contains all the states from numEps number of self-play games
                self.trainExamplesHistory.append(iterationTrainExamples)
                
            if len(self.trainExamplesHistory) > self.args['numItersForTrainExamplesHistory']:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file by calling saveTrainExamples method
            # The examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i-1) #save examples to self.args['checkpoint'] folder with given iteration name of i-1
            
            # shuffle examples before training
            #trainExamples is the list form of trainExamplesHistory. Note that trainExamplesHistory is a list of deques,
            #where each deque contains training examples. trainExamples gets rid of the deque, and instead puts all training
            #samples in a single list, shuffled
            trainExamples = []
            for e in self.trainExamplesHistory: #Each e is a deque
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args['checkpoint'], filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args) #the old MCTS with old neural net, before training p = previous
            
            #convert trainExamples into a format recognizable by Neural Network and train
            trainExamples = self.nnet.constructTraining(trainExamples)
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args) #the new neural network after training n = new
						
            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game, self.args, self.game_args) #note that Arena will pit pmcts with nmcts, and Game_args A and y will change constantly.
            pwins, nwins, draws = arena.playGames()

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins+nwins > 0 and float(nwins)/(pwins+nwins) < self.args['updateThreshold']
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')                

    def getCheckpointFile(self, iteration): #return a string which gives information about current checkpoint iteration
    #and file type (.tar)
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration): #save training examples (self.trainExamplesHistory to 
    	#args['checkpoint'] folder with name of self.getCheckpointFile with given iteration. 
        folder = self.args['checkpoint']
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
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



	