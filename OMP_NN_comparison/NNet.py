#Neural Network Wrapper for experiments. Use this to call different Neural Network structures.
import NetArch


class NNetWrapper():
    def __init__(self, args):
        self.naive_net = NetArch.naiveNet(args)
        self.OMPbootstrap_net = NetArch.bootstrap_Net(args)
        
        