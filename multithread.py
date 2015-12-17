from multiprocessing import Process, Queue
import scipy as sp
import svm_kernel


import matplotlib.pyplot as plt


def do_sum(q,l):
    # compute gradient over X and other X
    # feedback [index,gradient]
    q.put(sum(l))




def generate_train_params(N,X,W,Y):
    rn = sp.random.randint(N)
    rn2 = sp.random.randint(N)
    x1 = X[:,rn]
    x2 = X[:,rn2]
    w = W[rn2]
    y = Y[:,rn]
    pos = rn2
    return [rn,rn2,x1,x2,w,y,pos]




def main():

    k,kparam,reg,N,noise,X,y,iterations = svm_kernel.get_settings()
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
    for i in range(0,100000):
        rn,rn2,x1,x2,w,y,pos = generate_train_params(N,X,W,Y)
        argg = (q, x1, x2, y, w, pos, 1., .1,(svm_kernel.GaussianKernel, (1.)))
        p1 = Process(target=svm_kernel.fit_svm_kernel_double_random_one_update, args=argg)

        rn,rn2,x1,x2,w,y,pos = generate_train_params(N,X,W,Y)
        argg = (q, x1, x2, y, w, pos, 1., .1,(svm_kernel.GaussianKernel, (1.)))
        p2 = Process(target=svm_kernel.fit_svm_kernel_double_random_one_update, args=argg)

        rn,rn2,x1,x2,w,y,pos = generate_train_params(N,X,W,Y)
        argg = (q, x1, x2, y, w, pos, 1., .1,(svm_kernel.GaussianKernel, (1.)))
        p3 = Process(target=svm_kernel.fit_svm_kernel_double_random_one_update, args=argg)

        rn,rn2,x1,x2,w,y,pos = generate_train_params(N,X,W,Y)
        argg = (q, x1, x2, y, w, pos, 1., .1,(svm_kernel.GaussianKernel, (1.)))
        p4 = Process(target=svm_kernel.fit_svm_kernel_double_random_one_update, args=argg)

        rn,rn2,x1,x2,w,y,pos = generate_train_params(N,X,W,Y)
        argg = (q, x1, x2, y, w, pos, 1., .1,(svm_kernel.GaussianKernel, (1.)))
        p5 = Process(target=svm_kernel.fit_svm_kernel_double_random_one_update, args=argg)

        rn,rn2,x1,x2,w,y,pos = generate_train_params(N,X,W,Y)
        argg = (q, x1, x2, y, w, pos, 1., .1,(svm_kernel.GaussianKernel, (1.)))
        p6 = Process(target=svm_kernel.fit_svm_kernel_double_random_one_update, args=argg)

        rn,rn2,x1,x2,w,y,pos = generate_train_params(N,X,W,Y)
        argg = (q, x1, x2, y, w, pos, 1., .1,(svm_kernel.GaussianKernel, (1.)))
        p7 = Process(target=svm_kernel.fit_svm_kernel_double_random_one_update, args=argg)

        rn,rn2,x1,x2,w,y,pos = generate_train_params(N,X,W,Y)
        argg = (q, x1, x2, y, w, pos, 1., .1,(svm_kernel.GaussianKernel, (1.)))
        p8 = Process(target=svm_kernel.fit_svm_kernel_double_random_one_update, args=argg)


        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()
        p7.start()
        p8.start()
        r1 = q.get()
        r2 = q.get()
        r3 = q.get()
        r4 = q.get()
        r5 = q.get()
        r6 = q.get()
        r7 = q.get()
        r8 = q.get()
        discount = 1./(1. + i + N)/N
        #print discount
        W[r1[1]] -= discount * r1[0]
        W[r2[1]] -= discount * r2[0]
        W[r3[1]] -= discount * r3[0]
        W[r4[1]] -= discount * r4[0]
        W[r5[1]] -= discount * r5[0]
        W[r6[1]] -= discount * r6[0]
        W[r7[1]] -= discount * r7[0]
        W[r8[1]] -= discount * r8[0]

        # print temporary result and reiterate

        train_error = svm_kernel.test_svm(X, Y, W, (k, (kparam)))
        print train_error
        train_errors.append(train_error)
        svm_kernel.update_plot(train_errors)
        X_test,y_test = svm_kernel.make_data_xor(N, noise)
        print "test error:", svm_kernel.test_svm(X_test, y_test, W, (k, (kparam)))




if __name__=='__main__':
    main()



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