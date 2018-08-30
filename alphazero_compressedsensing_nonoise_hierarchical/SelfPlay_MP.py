#Self Play Multiprocessing class which allows self play games to be played concurrently across multiple CPU cores
#This object is called in Coach when we are generating many games of self-play. 
import multiprocessing as mp

class SelfPlay_MP():
    def __init__(self):
        pass