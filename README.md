# Reinforcement Learning for Compressed Sensing

# Introduction
The code in this repository contains a single GPU implementation of Alphazero for solving the Classic Compressed Sensing Problem. Given a fixed(or unfixed) matrix A of size m by n, m<<n, , and an observed vector y = Ax, where x is an unknown k-sparse vector, Alphazero attempts to iteratively pick the k support locations of x. 
# File Structure
"alphazero_compressedsensing_nonoise_hierarchical" and "alphazero_compressedsensing_nonoise_hierarchical_v2" are the current most stable versions. The difference between these two is that v2(the second file) contains functionality for hierarchical learning and code for manipulating the Monte Carlo Tree during training. 

"current_version/alphazero_compressedsensing_nonoise_hierarchical_v2" is the most recent code(which may contain bugs) which provides optimizations for GPU and CPU usage. The goal is to add functionality for parallel MCTS simulations and move linear algebra computations and Monte Carlo Tree Search over to the GPU. 

# General Usage
To start running the alphazero algorithm, simply run "python main.py" after navigating to the appropriate folder. All parameters in the algorithm are contained in main.py.

In each of these folders, the following crucial folders hold:
- source code for the general alphazero algorithm structure
- alphazero_testing: holds source for testing the algorithm
- compressed_sensing: holds source for the policy neural network and game rules
- fixed_sensing_matrix: either is empty or holds a single .npy file which the user provides if he/she wants to pre-load a sensing matrix A
- network_checkpoint: holds all trained policy value networks trained up to a specified number of iterations specified in main.py
- training_data: holds all generated training data during the course of the algorithm.
- skip_network: (not important to run the base Alphazero algorithm) needs to hold the weights and model of the neural network to skip depths in the MCTS tree if hierarchical learning is used. 

# Examples of Trained Policy/Value Networks 
Below, we include some examples of trained policy/value networks(without MCTS during inference/testing) versus popular compressed sensing algorithms. The first plot is the recovery accuracy for the 7 by 15 matrix on 6000 signals (1000 signals per sparsity on the x-axis) using 3 algorithms:

- l1-minimization (BP)
- Orthogonal Matching Pursuit (OMP)
- Alphazero (AlphaCS)

The second plot is the same as plot 1, except it is performed on the 15 by 50 matrix. Same as plot 1, 1000 signals per sparsity(for a total of 14,000 signals) are used for testing. 

![acc_plot_smallmatrices](https://user-images.githubusercontent.com/16004926/53453264-76c74200-39d8-11e9-92ec-c61c2c5ed046.png)
![acc_plot_smallmatrices2](https://user-images.githubusercontent.com/16004926/53453266-76c74200-39d8-11e9-9e68-0015b434b824.png)

# References
The starting point of the code was taken from the generalized alphazero algorithm in https://github.com/suragnair/alpha-zero-general, although the final code contained in this repository is now significantly altered. Those looking for a generalized alphazero algorithm should take a look at the link given.

# Contact
If there are any questions, please email me at sichen.zhong@stonybrook.edu
