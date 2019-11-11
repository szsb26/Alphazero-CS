import matplotlib
import matplotlib.pyplot as plt
import numpy as np

A = np.load('NN_to_acc.npy')
    
legend = []

for i in range(1, A.shape[0]):
    plt.plot(A[i,:])
    legend_label = str(i) + '-sparse'
    legend.append(legend_label)

plt.legend(legend, loc = 'upper left')    
plt.title('Accuracy of every NN in every iteration of Alphazero')
plt.ylabel('NN prediction accuracy')
plt.xlabel('NN at fixed iteration')
plt.savefig('accuracy_across_NN.png')