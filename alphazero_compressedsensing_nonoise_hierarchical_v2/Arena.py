import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
from MCTS import MCTS
import time

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, pnet, nnet, game, args, game_args): #game_args = arena_game_args. See Coach.py
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.
        """
        self.game = game #usually C.game if C is a Coach object. See Coach.py
        self.game_args = game_args #game_args here is going to be self.arena_game_args in Coach.learn()
        self.args = args
        self.player1 = pnet #lambda x: np.argmax(player1.getActionProb(x, temp=0))
        self.player2 = nnet #lambda x: np.argmax(player2.getActionProb(x, temp=0))
        self.player1wins = 0
        self.player2wins = 0
    

    def playGame(self, player1, player2, verbose=False): #player 1 and player 2 are lambda functions. Specifically, they are in the form of player1 = lambda x: np.argmax(pmcts.getActionProb(x, temp=0)), player2 = lambda x: np.argmax(nmcts.getActionProb(x, temp=0))
        """
        Executes one episode of a game. In the Compressed Sensing game, we see which
        neural network gets a reward which is smaller, and we pick the neural network model
        which chose the smallest sparsity. It is assumed Game_args.sensing_matrix and Game_args.obs_vector have already been generated.

        Returns: None
            self.player1wins += 1 if player 1 wins
            self.player2wins += 1 if player 2 wins
        """
        
        players = [player1, player2] 
        end_states = []
    
        #2 Games are played. P1 plays one game and P2 plays another game. The winner is the one who chooses has highest reward. 
        for i in range(2):
            state = self.game.getInitBoard(self.args, self.game_args) #Initialize the Game
            it = 0
            if verbose:
                print('Player ' + str(i+1) + ' is currently playing game...')
            while self.game.getGameEnded(state, self.args, self.game_args) == 0: #while we are not at a terminal state, continue playing CS game 
                it+=1
                #OUTPUT STATISTICS----------------------------
                if verbose: #default at false
                    print('Turn ', str(it))
                    print('current states action indices are: ' + str(state.action_indices))
                #----------------------------------------------
                #Determine the column chosen by player 1
                action = players[i](state)
                #FOR TESTING
                #print(action)
                #Determine if the chosen moves were valid or not. If not (valid[action]==0, raise exception)
                valids = self.game.getValidMoves(state)
                if valids[action]==0:
                    print(action)
                    assert valids[action]>0
                
                #Get the next state
                state = self.game.getNextState(state, action)
                
            end_states.append(state) #append the end state of P1 and P2. Note that states in this list already have termreward computed since self.game.getGameEnded was called.
                                     #end_states[0] is the end state achieved by P1, and end_states[1] is the end state achieved by P2. 
        
        
        #Determine whether pmcts or nmcts won. Otherwise, draw game.
        if end_states[0].termreward > end_states[1].termreward:
            self.player1wins += 1
        
        elif end_states[0].termreward < end_states[1].termreward:
            self.player2wins += 1
            
        else:
            self.player1wins += 1
            self.player2wins += 1
            
        #OUTPUT STATISTICS-------------------------------
        if verbose:
            print('Both Players have finished picking columns')
            print('Player 1 final reward: ' + str(end_states[0].termreward))
            print('Player 2 final reward: ' + str(end_states[1].termreward))
            print('Player 1 total wins: ' + str(self.player1wins))
            print('Player 2 total wins: ' + str(self.player2wins))
        #------------------------------------------------
        
        return None

    def playGames(self, verbose=False):
        """
        Plays num games in which in each game, A and y are re-generated 

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        
        #loop over number of games
        for i in range(self.args['arenaCompare']):
            if self.args['fixed_matrix'] == False:
                self.game_args.generateSensingMatrix(self.args['m'], self.args['n'], self.args['matrix_type']) 
            self.game_args.generateNewObsVec(self.args['x_type'], self.args['sparsity'])
            #Reinitialize the MCTS Tree for player 1 and 2. Note that every time A and y are reinitialized, the MCTS tree must be reinitialized because we are playing a new game. 
            #Note that 2*args['arenaCompare'] amount of MCTS trees are created in every call of Arena.playGames
            pmcts = MCTS(self.game, self.player1, self.args, self.game_args) #MCTS reinitialized, meaning that MCTS statistics (Nsa, Qsa, Ns, Ps, Es, Vs) are all reset. This is because choosing the same columns across games with different A's and y's count as different states even though we are choosing the same columns.
            nmcts = MCTS(self.game, self.player2, self.args, self.game_args)
            #Set the player1 and player2 lambda functions
            player1 = lambda x: np.argmax(pmcts.getActionProb(x, temp=0))
            player2 = lambda x: np.argmax(nmcts.getActionProb(x, temp=0))
            #play a game
            self.playGame(player1, player2, verbose)
            
        oneWon = self.player1wins
        twoWon = self.player2wins
        draws = self.player1wins + self.player2wins - self.args['arenaCompare']
        
        print('Player 1 total wins: ' + str(self.player1wins))
        print('Player 2 total wins: ' + str(self.player2wins))
        print('Total number of draws: ' + str(draws))

        return oneWon, twoWon, draws
