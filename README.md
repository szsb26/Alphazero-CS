# RL-for-CS

# Introduction
The code in this repository contains a single GPU implementation of Alphazero for solving the Classic Compressed Sensing Problem. Given a fixed(or unfixed) matrix A of size m by n, m<<n, , and an observed vector y = Ax, where x is an unknown k-sparse vector, Alphazero attempts to iteratively pick the k support locations of x. 
# File Explanation
"alphazero_compressedsensing_nonoise_hierarchical" and "alphazero_compressedsensing_nonoise_hierarchical_v2" are the current most stable versions. The difference between these two is that v2(the second file) contains functionality for hierarchical learning and code for manipulating the Monte Carlo Tree during training. 

"current_version/alphazero_compressedsensing_nonoise_hierarchical_v2" is the most recent code(which may contain bugs) which provides optimizations for GPU and CPU usage. The goal is to add functionality for parallel MCTS simulations and move linear algebra computations and Monte Carlo Tree Search over to the GPU. 
# Examples of Trained Policy/Value Networks 
Below, we include some examples of trained policy/value networks(without MCTS during inference/testing) versus popular compressed sensing algorithms. The first plot is the recovery accuracy for the 7 by 15 matrix on 6000 signals (1000 signals per sparsity on the x-axis) using 3 algorithms:

1)l1-minimization
2)Orthogonal Matching Pursuit
3)Alphazero

The second plot is the same as plot 1, except it is performed on the 15 by 50 matrix. Same as plot 1, 1000 signals per sparsity(for a total of 14,000 signals) are used for testing. 

![acc_plot_smallmatrices](https://user-images.githubusercontent.com/16004926/53453264-76c74200-39d8-11e9-92ec-c61c2c5ed046.png)
![acc_plot_smallmatrices2](https://user-images.githubusercontent.com/16004926/53453266-76c74200-39d8-11e9-9e68-0015b434b824.png)

# References
The starting point of the code was taken from the generalized alphazero algorithm in https://github.com/suragnair/alpha-zero-general, although the final code contained in this repository is now significantly altered. Those looking for a generalized alphazero algorithm should take a look at the link given.

# Contact
If there are any questions, please email me at sichen.zhong@stonybrook.edu
