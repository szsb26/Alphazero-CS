from Test_Class import Test

Test = Test()

#Testing Coach.executeEpisode()
#parameters args['tempThreshold']
#OUTPUT:returns a list of state objects, where state.pi_as, state.z, and state_feature_dic have all been computed
#FUNCTION:Play a full game to a terminal state and return a list of states as training samples. 

trainExamples = Test.coach.executeEpisode()
print('')
print('length of trainExamples is: ' + str(len(trainExamples)))

#Print some statistics about each state in trainExamples
for state in trainExamples:
    print('')
    print('Action Indices: ' + str(state.action_indices))
    print('Column_Indices: ' + str(state.col_indices))
    print('')
    print('Feature_dic: ')
    print('x_l2: ' + str(state.feature_dic['x_l2']))
    print('col_res_IP: ' + str(state.feature_dic['col_res_IP']))
    print('')
    print('state.p_as: ' + str(state.pi_as))
    print('state.z: ' + str(state.z))
    print('') 
    
print('The generated sparse vector is: ' + str(Test.game_args.sparse_vector))