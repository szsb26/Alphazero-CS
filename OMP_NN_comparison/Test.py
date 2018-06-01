#Test Class loads the the Testing environment. It also provides methods for graph drawing, etc...
#Takes in models which have been set to True in main and compares them. For Neural networks set to True in args, Test class
#already assumes that these Neural Networks have been trained. 
import matplotlib.pyplot as plt
from Gen_Data import GenData
from NNet import NNetWrapper

class Test():
    def __init__(self, args):
        self.Data = GenData()
        self.NNet = NNetWrapper(args)