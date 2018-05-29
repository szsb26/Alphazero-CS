from Test_Class import Test

#Construct Test Case
Test = Test()

#TESTING STATE METHODS AND NN PREDICTIONS
#----------------------------------------------------

#Test and output state self variables
game_start = Test.game.getInitBoard(Test.args, Test.game_args)
#print('The action indices are: ' + str(game_start.action_indices))
#print('')
#print('The column indices are: ' + str(game_start.col_indices))
#print('')

#Compute NN representation and test prediction
#game_start.compute_x_S_and_res(args, test_game_args)
#print('Feature Dictionary info: ')
#print(game_start.feature_dic)
#print('')
#game_start.converttoNNInput()
#print('NN_input data info: ')
#print(game_start.nn_input)
#print(len(game_start.nn_input))
#print(game_start.nn_input[0].shape)
#print(game_start.nn_input[1].shape)
#print('')

#TESTING SOME NNETWRAPPER METHODS
#----------------------------------------------------
#p_as, z = test_nnet.predict(game_start)
#print('The predicted probability dist is: ')
#print(p_as)
#print(p_as.shape)
#print('')
#print('The predicted reward is: ')
#print(z)
#print(z.shape)
#print('')


#TESTING MCTS OBJECT AND ITS METHODS
#----------------------------------------------------
print('Starting MCTS search on just the initial game state...')
print('')
print('The statistics of the initial game state are:')
print(game_start.action_indices)
print(game_start.col_indices)
probs = Test.MCTS.getActionProb(game_start) #50 numMCTSSims have been run to find the next move
print('')
print('Starting from the initial game state, the prob. dist. of the next move is:')
print(probs)
print('')
s = Test.game.stringRepresentation(game_start)
print('MCTS tree statistics for initial state, action 2')
print('self.Qsa[(s,2)]: ' + str(Test.MCTS.Qsa[(s, 2)]))
print('self.Nsa[(s,2)]: ' + str(Test.MCTS.Nsa[(s, 2)]))
print('self.Ns[s]: ' + str(Test.MCTS.Ns[s]))
print('self.Ps[s]: ' + str(Test.MCTS.Ps[s]))
print('self.Es[s]: ' + str(Test.MCTS.Es[s]))
print('self.Vs[s]: ' + str(Test.MCTS.Vs[s]))

print('Starting MCTS search on the state where action 2 was taken from initial game state')
print('')
print('The statistics of the state are:')
first_state = Test.game.getNextState(game_start, 2)
print(first_state.action_indices)
print(first_state.col_indices)
probs2 = Test.MCTS.getActionProb(first_state)
print('')
print('The prob. dist of the next move is:')
print(probs2)
print('')


#Print some MCTS object statistics
s1 = Test.game.stringRepresentation(first_state)
print('MCTS tree statistics for initial state, action 2')
print('self.Qsa[(s,2)]: ' + str(Test.MCTS.Qsa[(s, 2)]))
print('self.Nsa[(s,2)]: ' + str(Test.MCTS.Nsa[(s, 2)]))
print('self.Ns[s]: ' + str(Test.MCTS.Ns[s]))
print('self.Ps[s]: ' + str(Test.MCTS.Ps[s]))
print('self.Es[s]: ' + str(Test.MCTS.Es[s]))
print('self.Vs[s]: ' + str(Test.MCTS.Vs[s]))
print('self.Ns[s1]: ' + str(Test.MCTS.Ns[s1]))
print('self.Ps[s1]: ' + str(Test.MCTS.Ps[s1]))
print('self.Es[s1]: ' + str(Test.MCTS.Es[s1]))
print('self.Vs[s1]: ' + str(Test.MCTS.Vs[s1]))

print('the sparse vector x is: ' + str(Test.game_args.sparse_vector))

#print('After searching on the initial game state, the current self variables of the MCTS class are:')
#print('self.Qsa : ' + str(test_MCTS.Qsa))
#print('self.Nsa : ' + str(test_MCTS.Nsa))
#print('self.Ns : ' + str(test_MCTS.Ns))
#print('self.Ps : ' + str(test_MCTS.Ps))
#print('self.Es : ' + str(test_MCTS.Es))
#print('self.Vs : ' + str(test_MCTS.Vs))





