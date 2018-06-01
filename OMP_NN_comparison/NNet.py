#Neural Network Wrapper for experiments. Use this to call different Neural Network structures.
from NetArch import NetArch


class NNetWrapper():
    def __init__(self, args):
        self.netWrapper = NetArch()
        self.naive_net = self.netWrapper.naiveNet(args)
        self.OMPbootstrap_net = self.netWrapper.OMPbootstrap_Net(args)
    
    def naiveNN_train(self, training_data): #train the naive network with the given data
        pass
        
    def OMPbootstrapNN_train(self, training_data):  #train the naive network with the given data
        pass
    
    
    