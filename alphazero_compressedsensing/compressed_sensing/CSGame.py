from Game import Game
from CSState import State

class CSGame(Game):
    def __init__(self, A, y): #A and y are both assumed to be numpy arrays
        self.sensing_matrix = A
        self.obs_vector = y

    def getInitBoard(self): #Get the initial state at beginning of CS Game
    	col_size = A.shape[1]
    	Initial_State = CSState(np.ones(col_size))

    def getActionSize(self): #return number of all actions (equal to column size of A)
    	return A.shape[1]

    def getNextState(self, state, action): #input is a CSState object and an action(integer). Output
    											   #is the next state as a CSState object. 
        if state.col[action] == 1:
        	print ('Column already taken, invalid move')
        	return
        else:
        	nextstate_col_ind = np.array(state.col)
        	nextstate_col_ind[action] = 1
        	next_state = CSState(nextstate_col_ind)
        	return next_state 

    def getValidMoves(self, state): #input is a CSState object. Output is a binary numpy vector for valid moves,
    								#where b[i] = 1 implies ith column can be taken. 
    
        valid_moves = np.zeros(A.shape[1])
        for i in range(A.shape[1]):
        	if state.col[i] == 0:
        		valid_moves[i] = 1
		
		return valid_moves
		
    def getGameEnded(self, state): #check if state is a terminal state or not. If not terminal state, return 0. Otherwise,
    							   #return (negative of?) the sparsity. 
        """
        Input:
            board: current board

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw. 
        """
        pass
        

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        pass

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass
