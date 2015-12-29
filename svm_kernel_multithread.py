import threading
import time
import scipy as sp

import matplotlib.pyplot as plt

from svm_kernel import GaussianKernel, fit_svm_kernel_double_random_one_update, test_svm, make_plot_twoclass


class WildUpdater (threading.Thread):
    def __init__(self, threadID, name, counter, startpos, X,Y,its=100,eta=1.,C=.1,kernel=(GaussianKernel,(1.)),num_threads=8.0):

        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

        # to determine which chunk of training data it will use
        # we copy all datapoints, so it will use alot of memory
        # TODO: remove this is a (lazy) dirty hack.
        self.startpos = startpos
        # rest of update parameter
        self.X = X
        self.Y = Y
        self.eta = eta
        self.its = its
        self.C = C
        self.kernel = kernel
        self.num_threads = num_threads


    def run(self):
        global threadLock,W,X,Y,errors,updatecount

        D,N = self.X.shape[0],self.X.shape[1]
        Xlocal = sp.vstack((sp.ones((1,N)),self.X))

        self.endpos = int(self.startpos + 1/self.num_threads * N)

        # threadLock.acquire()
        # print "Starting " + self.name + " range:" + str(self.startpos) + " - " + str(self.endpos)
        # print "iterations:" + str(self.its) + " N:" +str(N)
        # threadLock.release()


        discount = 1.0
        max_its =  N
        for it in range(self.its):
            discount = self.eta/((updatecount+1.+max_its)/float(max_its)) # 0.99999
            #print discount
            rn = sp.random.randint(self.startpos,self.endpos)
            rn2 = sp.random.randint(self.startpos,self.endpos)
            G,pos = fit_svm_kernel_double_random_one_update(Xlocal[:,rn],Xlocal[:,rn2],Y[:,rn],W[rn2],rn2,self.kernel)

            # Get lock to synchronize threads
            W[pos] -= discount * G
            #threadLock.acquire()

            #compute error
            #add to error list
            if updatecount%N == 0:
                err = test_svm(self.X, self.Y, W, self.kernel)[0]
                errors.append([int(updatecount/N), err])

            # Free lock to release next thread
            updatecount += 1
            #threadLock.release()


def plot_lines(d1,d2):
	plt.plot(d1, d2, '-')
	plt.show()
	plt.pause(0.0000001) #Note this correction




updatecount = 0
threadLock = threading.Lock()
def fit_svm_kernel_double_random_threading(W_input,k, kparam, reg, N, noise, X, y, iterations, visualize=False):
    global threadLock,updatecount,Y,W,errors
    W = W_input.copy()

    threadLock = threading.Lock()
    updatecount = 0
    errors = []
    updatecount = 0
    #global threadLock,Y,X,W
    #k,kparam,reg,N,noise,X,y,iterations = get_settings()
    Y=y

    threads = []
    plt.ion()
    n_threads = 8.0

    iterations = int(iterations * N / n_threads )
    # print iterations * n_threads
    # Create new threads with different subsets of data
    startpos = 0
    thread1 = WildUpdater(1, "Thread-1", 1, startpos, X,y,its=iterations,eta=1.,C=.1,kernel=(k,(kparam)),num_threads=n_threads)
    startpos = int(N/n_threads)
    thread2 = WildUpdater(2, "Thread-2", 2, startpos, X,y,its=iterations,eta=1.,C=.1,kernel=(k,(kparam)),num_threads=n_threads)
    startpos = int(N/n_threads) * 2
    thread3 = WildUpdater(3, "Thread-3", 3, startpos, X,y,its=iterations,eta=1.,C=.1,kernel=(k,(kparam)),num_threads=n_threads)
    startpos = int(N/n_threads) * 3
    thread4 = WildUpdater(4, "Thread-4", 4, startpos, X,y,its=iterations,eta=1.,C=.1,kernel=(k,(kparam)),num_threads=n_threads)
    startpos = int(N/n_threads) * 4
    thread5 = WildUpdater(5, "Thread-5", 5, startpos, X,y,its=iterations,eta=1.,C=.1,kernel=(k,(kparam)),num_threads=n_threads)
    startpos = int(N/n_threads) * 5
    thread6 = WildUpdater(6, "Thread-6", 6, startpos, X,y,its=iterations,eta=1.,C=.1,kernel=(k,(kparam)),num_threads=n_threads)
    startpos = int(N/n_threads) * 6
    thread7 = WildUpdater(7, "Thread-7", 7, startpos, X,y,its=iterations,eta=1.,C=.1,kernel=(k,(kparam)),num_threads=n_threads)
    startpos = int(N/n_threads) * 7
    thread8 = WildUpdater(8, "Thread-8", 8, startpos, X,y,its=iterations,eta=1.,C=.1,kernel=(k,(kparam)),num_threads=n_threads)

    # Start new Threads
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    thread6.start()
    thread7.start()
    thread8.start()

    # Add threads to thread list
    threads.append(thread1)
    threads.append(thread2)
    threads.append(thread3)
    threads.append(thread4)
    threads.append(thread5)
    threads.append(thread6)
    threads.append(thread7)
    threads.append(thread8)



    if visualize:
        time.sleep(1)
        allrunning = True
        e1,e2 = zip(*errors)
        plot_lines(e1,e2)
        while(allrunning):

            num_dead = 0
            for t in threads:
                if t.isAlive() == False:
                    num_dead += 1
            if num_dead == len(threads):
                break
            #make_plot_twoclass(X,Y,W,(k,(kparam)))
            # plot shit
            e1,e2 = zip(*errors)
            plot_lines(e1,e2)
    # Wait for all threads to complete
    for t in threads:
        t.join()
    # print "Exiting Main Thread"
    return W,errors
'''
for i in range(0,100):
    errors = []
    updatecount = 0
    test_threading()
'''