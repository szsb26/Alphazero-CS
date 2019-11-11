import sys
sys.path.append("..")
from pytorch_classification.utils import Bar, AverageMeter
import numpy as np
import time

length_list = 50

bar = Bar('Test Bar', max = length_list)
test_list = list(range(1, length_list))
eps_completed = 0
eps = 0


while test_list:
    a = np.random.binomial(1, 0.5)
    b = np.random.binomial(1, 0.5)
    
    if a == 1 and b == 1:
        test_list.pop()
        test_list.pop()
        
        eps_completed += 2
        bar.suffix  = '({eps}/{maxeps})'.format(eps = eps_completed, maxeps=length_list)
        bar.next()
        #bar.suffix  = '({eps}/{maxeps})'.format(eps = eps_completed, maxeps=length_list)
        time.sleep(3)
    
    elif a == 0 and b == 0:
        test_list.pop()
        
        eps_completed += 1
        #bar.suffix only controls the suffix output only
        bar.suffix  = '({eps}/{maxeps})'.format(eps = eps_completed, maxeps=length_list)
        #bar.next() controls the drawing of the completion bar only
        bar.next()
        #bar.suffix  = '({eps}/{maxeps})'.format(eps = eps_completed, maxeps=length_list)
        time.sleep(3)
    

bar.finish()