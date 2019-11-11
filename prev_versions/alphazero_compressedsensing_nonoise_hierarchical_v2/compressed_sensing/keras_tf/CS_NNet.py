from keras.models import Model
from keras.layers import Input, Dense, concatenate
from keras.optimizers import Adam
from keras import regularizers
from keras import backend
from numpy.random import seed
#seed(1)


class NetArch():
    def __init__(self, args):
    #INPUTS:
    #args(args is from main.py) and current game
    #OUTPUT:
    #the Neural Network architecture model. Look to pg 78 in notes for a pictorial representation of NN architecture.
        
        self.args = args
    
        #Create the input and first hidden layer based on which features are initialized in args
        Inputs = []
        HL1 = []
        #Depending on which features are given in args, create the input size of NN
        if self.args['x_l2']:
            xl2_input1 = Input(shape = (self.args['n'],)) #corresponds to a combination of feature a and b (pg 78), tensor object
            Inputs.append(xl2_input1)
        if self.args['lambda']:
            colresIP_input2 = Input(shape = (self.args['n'],)) #corresponds to feature b1 in notes  (pg 78), tensor object
            Inputs.append(colresIP_input2)
        
        
        #Build first layer (not fully connected) by iteratively building sets of neurons corresponding to an input feature. 
        #Concatenate all of these neurons to form the first hidden layer x. 
        for input in Inputs:
             HL1_set = Dense(self.args['neurons_per_layer'], activation = 'relu')(input)
             HL1.append(HL1_set)
        
        x = concatenate(HL1, axis = -1)
    
        #Define the subsequent fully connected hidden layers depending on self.args['num_layers']
        for i in range(self.args['num_layers']-1):
            x = Dense(self.args['neurons_per_layer'], activation = 'relu')(x)
            
        #Define the two outputs, where activation is softmax and identity, since reward in mcts is
        #sparsity + l2 regularization term.
        self.p_as = Dense(self.args['n']+1, activation = 'softmax', name = 'p_as')(x)#note the + 1 is for choosing the stopping action.
        self.v = Dense(1, name = 'v')(x)
        
        #FOR TESTING--------------------------------------------
        #self.v = Dense(1, activation = 'tanh', name = 'v')(x)
        #END TESTING--------------------------------------------
        
        self.model = Model(inputs = Inputs, outputs = [self.p_as, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], metrics=['accuracy'], optimizer=Adam(self.args['lr']))

class NetArch1(): # A 2 hidden layer neural network with only x_l2 as features and v, pas as output
    def __init__(self, args):
        
        self.args = args
        
        if self.args['x_l2']:
            xl2_input1 = Input(shape = (self.args['n'],))
        else:
            print('Error: no features were activated')
            return
            
        #Define hidden layer 1 of neural network
        HL1 = Dense(self.args['m'], activation = 'relu')(xl2_input1)
        #Define second hidden layer
        HL2 = Dense(self.args['neurons_per_layer'], activation = 'relu')(HL1)
        #Define output layers
        self.p_as = Dense(self.args['n']+1, activation = 'softmax', name = 'p_as')(HL2)
        self.v = Dense(1, name = 'v')(HL2)
        
        self.model = Model(inputs = xl2_input1, outputs = [self.p_as, self.v])
        self.model.compile(loss = ['categorical_crossentropy', 'mean_squared_error'], metrics = ['accuracy'], optimizer = Adam(self.args['lr']))
            
class NetArch2(): # A single layer fully connected neural network
    def __init__(self, args):
    #INPUTS:
    #args(args is from main.py) and current game
    #OUTPUT:
    #the Neural Network architecture model. Look to pg 78 in notes for a pictorial representation of NN architecture.
        
        self.args = args
    
        #Create the input and first hidden layer based on which features are initialized in args
        Inputs = []
        HL1 = []
        #Depending on which features are given in args, create the input size of NN
        if self.args['x_l2']:
            xl2_input1 = Input(shape = (self.args['n'],)) #corresponds to a combination of feature a and b (pg 78), tensor object
            Inputs.append(xl2_input1)
        if self.args['lambda']:
            colresIP_input2 = Input(shape = (self.args['n'],)) #corresponds to feature b1 in notes  (pg 78), tensor object
            Inputs.append(colresIP_input2)
        
        final_inputs = concatenate(Inputs, axis = -1)
        #hidden_layer = Dense(self.args['neurons_per_layer'], activation = 'relu', activity_regularizer = regularizers.l2(0.01), kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01))(final_inputs)
        #activation in the hidden layer below has very large values, when A is bernoulli. This is because the solution x_S to the l2 minimization problem has very large entries!!!
        hidden_layer = Dense(self.args['neurons_per_layer'], activation = 'relu')(final_inputs)
            
        #Define the two outputs, where activation is softmax and identity, since reward in mcts is
        #sparsity + l2 regularization term.
        self.p_as = Dense(self.args['n']+1, activation = 'softmax', name = 'p_as')(hidden_layer)#note the + 1 is for choosing the stopping action.
        
        #Note that v is usually between 0 and -1, so we use negative sigmoid here as an activation in the output layer. 
        #def neg_sigmoid(x):
            #return(-1*backend.sigmoid(x))
        
        #self.v = Dense(1, activation = 'neg_sigmoid', name = 'v')(hidden_layer)
        #self.v = Dense(1, activation = 'tanh', name = 'v')(hidden_layer)
        self.v = Dense(1, name = 'v')(hidden_layer)
        
        self.model = Model(inputs = Inputs, outputs = [self.p_as, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], metrics=['accuracy'], optimizer=Adam(self.args['lr']))
        
