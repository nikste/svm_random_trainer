import threading
from multiprocessing import Process, Queue
import scipy as sp
import svm_kernel
import numpy as np

import matplotlib.pyplot as plt



def generate_train_params(N,X,W,Y):
    rn = sp.random.randint(N)
    rn2 = sp.random.randint(N)
    x1 = X[:,rn]
    x2 = X[:,rn2]
    w = W[rn2]
    y = Y[:,rn]
    pos = rn2
    return [rn,rn2,x1,x2,w,y,pos]


'''
"single node in cluster"
'''
class Iteration_thread(threading.Thread):


    def __init__(self, threadno, iterations, X, Y, excerpt_list):
        super(Iteration_thread, self).__init__()
        self.threadno = threadno
        self.iterations = iterations
        self.X = X
        self.Y = Y
        self.discount = 1.0
        self.excerpt_list = excerpt_list
        self.N = len(excerpt_list)

    def send_gradient(self,update_message):
        global W
        W[update_message[1]] -= self.discount * update_message[0]

    def compute_gradient(self):

        # compute gradient
        rn,rn2,x1,x2,w,y,pos = self.generate_training_params()
        argg = (x1, x2, y, w, pos, 1., .1,(svm_kernel.GaussianKernel, (1.)))
        w_up,pos = svm_kernel.fit_svm_kernel_double_random_one_update(x1, x2, y, w, pos, 1., .1,(svm_kernel.GaussianKernel, (1.)))

        # update model
        self.send_gradient([w_up,pos])

    def generate_training_params(self):
        global W
        rn = self.excerpt_list[sp.random.randint(self.N)]
        rn2 = self.excerpt_list[sp.random.randint(self.N)]
        x1 = self.X[:,rn]
        x2 = self.X[:,rn2]
        w = W[rn2]
        y = self.Y[:,rn]
        pos = rn2
        return [rn,rn2,x1,x2,w,y,pos]

    def run(self):
         for i in range(0,self.iterations):
             print "t:",self.threadno,"it:",i
             self.compute_gradient()


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


finished = False
W = []

def main_threads():
    global W
    k,kparam,reg,N,noise,X,y,iterations,num_parallelprocesses = svm_kernel.get_settings()
    Y = y

    D,N = X.shape[0],X.shape[1]
    X = sp.vstack((sp.ones((1,N)),X))
    W = sp.randn(N)


    plt.ion() ## Note this correction
    fig=plt.figure()

    perms = np.arange(N)
    perms = np.random.permutation(perms)

    # distribute data
    num_threads = 4
    assert(N % num_threads == 0)
    perms_threads = (list(chunks(perms, N/num_threads)))

    print len(perms_threads)
    iterations = N * 100000
    excerpt_list = perms_threads[0]
    threads = []
    threadno = 0
    for els in perms_threads:
        print "starting thread:",threadno
        t = Iteration_thread(threadno, iterations, X, Y, els)
        threadno += 1
        t.start()
        threads.append(t)


    # c = Iteration_control_thread(local_w, iterations, X, Y, excerpt_list)
    # c.start()
    train_errors = []
    for i in range(100000):
        for l in threads:
            if not l.isAlive():
                pass
            train_error,point_error = svm_kernel.test_svm(X, Y, W, (svm_kernel.GaussianKernel, (1.)))
            print "train error:",train_error,point_error
            train_errors.append(train_error)

            svm_kernel.update_plot(train_errors)
            if finished:
                break

    for els in threads:
        els.join()
    # c.join()
    print "yea"


def main_queue():

    k,kparam,reg,N,noise,X,y,iterations,num_parallelprocesses = svm_kernel.get_settings()
    Y = y

    D,N = X.shape[0],X.shape[1]
    X = sp.vstack((sp.ones((1,N)),X))
    W = sp.randn(N)
    rn = sp.random.randint(N)
    rn2 = sp.random.randint(N)

    plt.ion() ## Note this correction
    fig=plt.figure()

    train_errors = []
    q = Queue()



    # put iteration for loop here
    discount = 1
    for i in range(0,iterations):
        for j in range(0,N/num_parallelprocesses):
            ps = []
            for p in range(0,num_parallelprocesses):
                rn,rn2,x1,x2,w,y,pos = generate_train_params(N,X,W,Y)
                argg = (q, x1, x2, y, w, pos, 1., .1,(svm_kernel.GaussianKernel, (1.)))
                ps.append(Process(target=svm_kernel.fit_svm_kernel_double_random_one_update, args=argg))
                ps[-1].start()
            # rn,rn2,x1,x2,w,y,pos = generate_train_params(N,X,W,Y)
            # argg = (q, x1, x2, y, w, pos, 1., .1,(svm_kernel.GaussianKernel, (1.)))
            # p2 = Process(target=svm_kernel.fit_svm_kernel_double_random_one_update, args=argg)
            #
            # rn,rn2,x1,x2,w,y,pos = generate_train_params(N,X,W,Y)
            # argg = (q, x1, x2, y, w, pos, 1., .1,(svm_kernel.GaussianKernel, (1.)))
            # p3 = Process(target=svm_kernel.fit_svm_kernel_double_random_one_update, args=argg)
            #
            # rn,rn2,x1,x2,w,y,pos = generate_train_params(N,X,W,Y)
            # argg = (q, x1, x2, y, w, pos, 1., .1,(svm_kernel.GaussianKernel, (1.)))
            # p4 = Process(target=svm_kernel.fit_svm_kernel_double_random_one_update, args=argg)
            #
            # rn,rn2,x1,x2,w,y,pos = generate_train_params(N,X,W,Y)
            # argg = (q, x1, x2, y, w, pos, 1., .1,(svm_kernel.GaussianKernel, (1.)))
            # p5 = Process(target=svm_kernel.fit_svm_kernel_double_random_one_update, args=argg)
            #
            # rn,rn2,x1,x2,w,y,pos = generate_train_params(N,X,W,Y)
            # argg = (q, x1, x2, y, w, pos, 1., .1,(svm_kernel.GaussianKernel, (1.)))
            # p6 = Process(target=svm_kernel.fit_svm_kernel_double_random_one_update, args=argg)
            #
            # rn,rn2,x1,x2,w,y,pos = generate_train_params(N,X,W,Y)
            # argg = (q, x1, x2, y, w, pos, 1., .1,(svm_kernel.GaussianKernel, (1.)))
            # p7 = Process(target=svm_kernel.fit_svm_kernel_double_random_one_update, args=argg)
            #
            # rn,rn2,x1,x2,w,y,pos = generate_train_params(N,X,W,Y)
            # argg = (q, x1, x2, y, w, pos, 1., .1,(svm_kernel.GaussianKernel, (1.)))
            # p8 = Process(target=svm_kernel.fit_svm_kernel_double_random_one_update, args=argg)


            # p1.start()
            # p2.start()
            # p3.start()
            # p4.start()
            # p5.start()
            # p6.start()
            # p7.start()
            # p8.start()
            ress = []
            for r in range(0,num_parallelprocesses):
                ress.append(q.get())
            # r1 = q.get()
            # r2 = q.get()
            # r3 = q.get()
            # r4 = q.get()
            # r5 = q.get()
            # r6 = q.get()
            # r7 = q.get()
            # r8 = q.get()
            for r in ress:
                W[r[1]] -= discount * r[0]

        discount = 1./(1.0 + i)
        print "discount",discount

        train_error,point_error = svm_kernel.test_svm(X, Y, W, (k, (kparam)))
        print "train error:",train_error,point_error
        train_errors.append(train_error)

        svm_kernel.update_plot(train_errors)
        X_test,y_test = svm_kernel.make_data_xor(N, noise)
        print "test error:", svm_kernel.test_svm(X_test, y_test, W, (k, (kparam)))




if __name__=='__main__':
    main_threads()
    #main_queue()



'''
class foo():
    i = 0
    def do_task(self):
        self.i += 1
        print "hey you!hey! I already called you:",self.i,"times!"


def work(foo):
    foo.do_task()

from multiprocessing import Pool

pool = Pool()
f1 = foo()
f2 = foo()
f3 = foo()
flist = [f1,f2,f3]
pool.map(work, flist)
pool.close()
pool.join()
'''