import matplotlib
import matplotlib.pyplot as plt
import pickle

#plot trainHistoryDict
#plot validation acc
#bootstrap_val_acc = [0,1,0.71,0.32,0.1,0.01,0.005]
#OMP_val_acc = [0,1,0.72,0.365,0.115,0.02,0.005]


with open('trainHistoryDict', 'rb') as f:
    trainHistoryDict = pickle.load(f)


plt.plot(trainHistoryDict['p_as_acc'])
plt.plot(trainHistoryDict['val_p_as_acc'])
plt.legend(['train', 'test'], loc='upper left')
plt.title('model p_as accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('training_plot' + '/pas_acc.png')
#Clear the plot
plt.gcf().clear()
#plot and save loss graph for p_as
plt.plot(trainHistoryDict['p_as_loss'])
plt.plot(trainHistoryDict['val_p_as_loss'])
plt.legend(['train', 'test'], loc='upper left')
plt.title('model p_as loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('training_plot' + '/pas_loss.png') 
#Clear the plot
plt.gcf().clear()
#plot and save val acc for p_as
#plt.plot(bootstrap_val_acc)
#plt.plot(OMP_val_acc)
#plt.title('bootstrapNet vs OMPNet')
#plt.ylabel('accuracy')
#plt.xlabel('sparsity')
#plt.savefig('bootvsOMP.png')
#Clear the plot
#plt.gcf().clear()
#plot v_loss and val_v_loss
plt.plot(trainHistoryDict['v_loss'])
plt.plot(trainHistoryDict['val_v_loss'])
plt.legend(['train', 'test'], loc='upper left')
plt.title('model v loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('training_plot' + '/v_loss.png') 