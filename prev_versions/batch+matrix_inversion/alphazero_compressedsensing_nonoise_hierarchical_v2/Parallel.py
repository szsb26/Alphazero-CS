from multiprocessing import Process


#This class contains methods which handles parallel searching over a list of MCTS objects. 
class Parallel():

    def __init__ (self, args, nnet):
        self.args = args
        self.nnet = nnet
        
        
    def getActionProbs (self, MCTS_States_list):
        #job of this function is to take in a list of (MCTS object, [States list]) pairs, and for each pair, output
        #the probability distribution of the next move by running numMCTSsims number of mass_searches. Once numMCTSsims
        #number of mass_searches have been run, we look at each MCTS object and return the probability distribution of the next
        #move. 
        
        pass
        
    def parallel_search (self, MCTS_States_list):
        #search_search performs a single parallel_search(simulation) on every MCTS object in MCTS_States_list. A single simulation on a single
        #MCTS tree traverses from the root to a leaf. Once the traversal arrives at a leaf, the NN is called to return the prob. dist and v.
        #p_as is stored at the leaf node, and the leaf node is also expanded by constructing its neighbors. Finally, the edge weights we traversed
        #in the search are updated by using v from the NN. 
        
        #search_all conducts a parallel search by waiting until every MCTS search job has reached a leaf(unless the MCTS search ended up in a terminal node instead) in their 
        #corresponding trees. It then pools the NN queries into a single query. Each search job then uses the output of this single NN query to continue with expanding the leaf nodes and updating 
        #edge weights, which completes this single search on a single tree.
        
        pass
        
        