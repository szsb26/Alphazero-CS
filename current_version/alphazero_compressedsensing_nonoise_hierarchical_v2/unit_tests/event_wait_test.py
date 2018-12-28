import threading
import multiprocessing
import time


class Test_Object():
    def __init__(self, num):
        self.val = num


class Test():

    def __init__(self):
        self.exit_flag = threading.Event()
        self.nonsense = 0
        self.halted_threads = 0


    def square(self, test_object):
        
        #halt the algorithm. Note that once wait is called, set() must be called elsewhere to unlock the thread/process
        #One should always call the Test.square function in a Thread/Process object. Otherwise, the entire algorithm stalls if there is
        #just one interpreter.
        #self.exit_flag.wait() essentially waits until the exit_flag signal has been set to true via exit_flag.set()
        self.halted_threads += 1
        self.exit_flag.wait()
        
        
        #After released from waiting divide by itself
        test_object.val = test_object.val**2
        
        #Testing how to reset the exit_flag to false. IOW, we test clear() and wait(). clear() resets the exit_flag to False
        
        #self.exit_flag.clear()
        #self.exit_flag.wait()
        
        test_object.val = 1
        
    def compute_nonsense(self):
        
        for i in range(1000000):
            self.nonsense += 1
        
    
#TEST1(multiprocessing tests)----------------------------------------    
#Test1 = Test()

#Test1.square(5)

#Calling multiprocessing is equivalent to opening up an entire new instance of python, so all declared objects and variables
#do not carry over. In other words, in each of child processes p1 and p2, they are receiving a copy of Test1 and Test2 respectively.
#Hence the original Test1 and Test2 self variables do not update. 



#input_p1 = multiprocessing.Queue() #used for retrieving output from processes since by above, processes dont share memory.
#p1 = multiprocessing.Process(target = Test1.square, args = [6])


#p1.start()
#p1.join()

 
#print('TEST1')
#print(Test1.value)
#TEST2(threading tests)----------------------------------------------
#exit_flag is only in Test object.
Test = Test()
test_object1 = Test_Object(5)
test_object2 = Test_Object(6)

test_objects = [test_object1, test_object2]
#Define threading operation

threads = []

start = time.time()
for test_object in test_objects:
    t = threading.Thread(target = Test.square, args = (test_object,))
    threads.append(t)
    #The first thread t which reaches exit_flag.wait() will stop
    t.start()
    #Putting t.join() below will cause the program to halt, since t will never finish due to the exit_flag.wait() command above.
    #t.join()

while Test.halted_threads < 2:
    pass

#for every test_object in test_objects, test_object.val should BE THE SAME DUE TO THE exit_flag.wait() in square method
i = 1
print('Output 1')
for test_object in test_objects: 
    print('test_object value' + str(i) + ':', test_object.val)
    i += 1

#Set flag to true to stop the wait in square function. By commenting this out, the threads
#t above should all hang and the program should stall.
#When we activate the below line, the exit_flag is set to true, 
#and since all threads t rely on this single object Test, all the threads are activated and resumed

Test.exit_flag.set()

#time.sleep(5)
#Test.exit_flag.set()

#Below command waits till all threads t in threads are finished before reading more code below.
#----------------------------
#for t in threads:
    #t.join()

#We can also just wait 5 seconds, in which then the threads should have finished running.
#time.sleep(5)
#----------------------------
#Due to the flag being set to true in the line above, now for each test_object in
#test_objects, test_object.val should be divided by two.
i = 0
print('Output 2')
for test_object in test_objects: 
    print('test_object value' + str(i) + ':', test_object.val)
    i += 1
    

    




    
    


    
    
