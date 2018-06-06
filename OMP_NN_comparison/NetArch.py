from keras.models import Sequential, Model
from keras.layers import Input, Dense, concatenate
from keras.optimizers import Adam
from keras import regularizers


class NetArch():
    def naiveNet(self, args):
        model = Sequential()
        model.add(Dense(args['naiveNN_hidden_neurons'], input_dim = args['m'], activation = 'relu', activity_regularizer = regularizers.l1(0.01)))
        model.add(Dense(args['n'], activation = 'sigmoid'))
        adam_custom = Adam(args['naiveNN_lr'])
        model.compile(loss = 'binary_crossentropy', optimizer = adam_custom, metrics = ['accuracy'])
        
        return model
    
    
    def OMPbootstrap_Net(self, args):
        #Create the input and first hidden layer based on which features are initialized in args
        Inputs = []
        HL1 = []
        #Create input layer
        xl2_input1 = Input(shape = (args['n'],)) #tensor object
        Inputs.append(xl2_input1)
        colresIP_input2 = Input(shape = (args['n'],)) #tensor object
        Inputs.append(colresIP_input2)
        
        #Build first layer (not fully connected) by iteratively building sets of neurons corresponding to an input feature. 
        #Concatenate all of these neurons to form the first hidden layer x. 
        for input in Inputs:
             HL1_set = Dense(args['OMPbootstrap_neurons_per_layer'], activation = 'relu')(input)
             HL1.append(HL1_set)
        
        x = concatenate(HL1, axis = -1)
    
        for i in range(args['OMPbootstrap_num_layers']-1):
            x = Dense(args['OMPbootstrap_neurons_per_layer'], activation = 'relu')(x)
            
        #Define the output
        p_as = Dense(args['n'], activation = 'softmax', name = 'p_as')(x) 
        
        model = Model(inputs = Inputs, outputs = [p_as])
        model.compile(loss=['categorical_crossentropy'], metrics=['accuracy'], optimizer=Adam(args['OMPbootstrap_lr']))
        
        return model

    