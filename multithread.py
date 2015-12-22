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


    def __init__(self, threadno, iterations, X, Y, excerpt_list, discount):
        super(Iteration_thread, self).__init__()
        self.threadno = threadno
        self.iterations = iterations
        self.X = X
        self.Y = Y
        self.discount = 1.0
        self.excerpt_list = excerpt_list
        self.N = len(excerpt_list)
        self.discount = discount

    def send_gradient(self,update_message):
        global W
        self.discount *= 0.99999
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
        global update_counter
        global locked
        local_counter = 0
        for i in range(0,self.iterations):

            #if(i%10000==0):
            #    print "t:",self.threadno,"it:",i," dis:",self.discount
            self.compute_gradient()

            local_counter += 1
            if( not locked):
                locked = True
                #print "thread:",self.threadno,"reporting, adding",local_counter
                update_counter += local_counter
                locked = False
                local_counter = 0


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

locked = False
finished = False
W = []
update_counter = 0
def main_threads(meta_it):
    global W
    global update_counter
    global locked

    serve_figure = False

    k,kparam,reg,N,noise,X,y,iterations,num_parallelprocesses,num_threads = svm_kernel.get_settings()
    Y = y

    D,N = X.shape[0],X.shape[1]
    X = sp.vstack((sp.ones((1,N)),X))
    W = sp.randn(N)

    if serve_figure:
        plt.ion() ## Note this correction
        fig=plt.figure()

    perms = np.arange(N)
    perms = np.random.permutation(perms)

    # distribute data

    assert(N % num_threads == 0)
    perms_threads = (list(chunks(perms, N/num_threads)))

    print len(perms_threads)
    print N
    print iterations
    threads = []
    threadno = 0

    train_errors = []
    iteration_counter = 0

    iteration_counter += 1
    train_error,point_error = svm_kernel.test_svm(X, Y, W, (svm_kernel.GaussianKernel, (1.)))
    print iteration_counter,"train error:",train_error,point_error
    train_errors.append(train_error)

    ##svm_kernel.update_plot(train_errors)


    discount = 1.0
    for i in range(0,iterations):
        threads = []
        # start threads
        for els in perms_threads:
            print "starting thread:",threadno
            print discount
            t = Iteration_thread(threadno, N/num_threads, X, Y, els,discount)
            threadno += 1
            t.start()
            threads.append(t)

        discount *= pow(0.99999, (i + 1) * N/num_threads)

        for els in threads:
            els.join()
        # test result
        train_error,point_error = svm_kernel.test_svm(X, Y, W, (svm_kernel.GaussianKernel, (1.)))
        print "///////////////////////",i,"train error:",train_error,point_error," counter="
        train_errors.append(train_error)
        if serve_figure:
            svm_kernel.update_plot(train_errors)
    print "done"
    '''
    still_need_to_decrease_update_counter = False
    while(not finished):
        num_not_alive = 0
        for l in threads:
            if not l.isAlive():
                num_not_alive += 1
        if num_not_alive == len(threads):
            finished = True

        if still_need_to_decrease_update_counter:
            if not locked:
                locked = True
                print "update_counter before: ",update_counter
                update_counter -= N
                print "update_counter after: ", update_counter
                still_need_to_decrease_update_counter = False
                locked = False
            else:
                still_need_to_decrease_update_counter = True

        if update_counter > N:
            if not locked:
                print "not locked"
                locked = True
                print "update_counter before: ",update_counter
                update_counter -= N
                print "update_counter after: ",update_counter
                still_need_to_decrease_update_counter = False
                locked = False
            else:
                print "locked"
                still_need_to_decrease_update_counter = True

            iteration_counter += 1
            train_error,point_error = svm_kernel.test_svm(X, Y, W, (svm_kernel.GaussianKernel, (1.)))
            print "///////////////////////",iteration_counter,"train error:",train_error,point_error," counter=",update_counter," N is",N
            train_errors.append(train_error)

            ##svm_kernel.update_plot(train_errors)
        if finished:
            break

    for els in threads:
        els.join()
    # c.join()
    '''
    print "yea"

    # save file
    f = open("./res/double_random_output_geg" + str(reg) + "_N" + str(N) + "_noise" + str(noise) + "_iterations" + str(iterations) + "_iterative_" + str(meta_it),"w")
    for l in train_errors:
        f.write(str(l) + "\n")
    f.close()


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

            ress = []
            for r in range(0,num_parallelprocesses):
                ress.append(q.get())

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
    j = 9
    mini = 10 * (j - 1)
    maxi = 10 * j
    for i in range(mini,maxi):
        print("///////////////////////////////////" + str(i) +"////////////////////////////////////////")
        main_threads(i)
    #main_queue()
