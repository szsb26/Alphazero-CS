from Test_Class import Test
from collections import deque
from random import randint
from keras.optimizers import Adam
#Load the Default Test Environment
Test = Test()

#Testing Coach.executeEpisode()
#parameters used: args['tempThreshold']
#OUTPUT:returns a list of state objects, where state.pi_as, state.z, and state_feature_dic have all been computed
#FUNCTION:Play a full game to a terminal state and return a list of states as training samples. 
if False:
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
    print ('')
    print('The sensing matrix is: ' + str(Test.game_args.sensing_matrix))

#Testing Coach.learn()
#parameters used: 
#args['numIters'] - number of iterations of alphazero algorithm. An iteration consists of self-play and retraining NN
#args['numEps'] - number of games generated/ number of times executeEpisode() is called
#args['numItersForTrainExamplesHistory']
#args['maxlenofQueue']
#args['m'], args['n'], args['matrix_type'], args['x_type'], args['sparsity']
#args['checkpoint']
#args['load_folder_(folder)'], args['load_folder_(filename)']

#Testing first part of Coach.learn().
#1)Testing Coach.getCheckpointFile, Coach.saveTrainExamples(self, iteration) and Coach.loadTrainExamples(self)
if False:
    trainExamples = Test.coach.executeEpisode()
    iterationTrainExamples = deque([], maxlen=Test.args['maxlenOfQueue'])
    iterationTrainExamples += trainExamples
    Test.coach.trainExamplesHistory.append(iterationTrainExamples)
    #Print out action_indices of a random state for verification later after loading 
    a = randint(0,len(Test.coach.trainExamplesHistory)-1)
    b = randint(0,len(Test.coach.trainExamplesHistory[a])-1)
    print('')
    print(Test.coach.trainExamplesHistory[a][b].action_indices)
    print('')
    Test.coach.saveTrainExamples(1)#save Test.coach.trainExamplesHistory to args['checkpoint'], a folder directory with name of 'checkpoint_' + str(iteration) + '.pth.tar' + '.examples'
    Test.coach.trainExamplesHistory = [] #Reinitialize Test.coach.trainExamplesHistory to [].
    Test.args['load_folder_(filename)'] = 'checkpoint_1.pth.tar'
    Test.coach.loadTrainExamples()#load training examples with filepath of args['load_folder_(folder)'] and name of args['load_folder_(filename)'] + '.examples' into self.trainExamplesHistory.  
    #Verify that the TrainExamples are loaded correctly
    print(Test.coach.trainExamplesHistory[a][b].action_indices)
    
#2)Testing Neural Network Training
if True:
    trainExamples = Test.coach.executeEpisode()
    conv_training = Test.coach.nnet.constructTraining(trainExamples)
    Test.coach.nnet.train(conv_training[0], conv_training[1])
    Test.coach.nnet.save_checkpoint(folder=Test.args['checkpoint'], filename = 'temp.pth.tar') #testing nnet.save_checkpoint functionality
    Test.coach.nnet.load_checkpoint(folder=Test.args['checkpoint'], filename = 'temp.pth.tar') #testing the nnet.load_checkpoint functionality
    Test.coach.nnet.nnet.model.compile(loss=['categorical_crossentropy','mean_squared_error'], metrics=['accuracy'], optimizer=Adam(Test.args['lr'])) #NEED TO COMPILE MODEL AFTER LOADING
    Test.coach.nnet.train(conv_training[0], conv_training[1])
    