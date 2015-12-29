import os

import svm_kernel
import matplotlib.pyplot as plt
import numpy as np
from svm_kernel_multithread import fit_svm_kernel_double_random_threading

'''
draws plot with new data
'''
def update_plot(data):
	plt.plot(data)
	plt.show()
	plt.pause(0.0000001) #Note this correction

'''
generate experiment settings
'''
def get_settings():
    k = svm_kernel.GaussianKernel
    kparam = 1.
    reg = .001
    iterations = 100#100

    N = 100
    noise = .25#.1
    X,y = svm_kernel.make_data_xor(N, noise)

    return [k,kparam,reg,N,noise,X,y,iterations]


def save_results(filename,data):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    f = open(filename,'w')
    for el in data:
        f.write(str(el)+"\n")
    f.close()

'''
runs standard kernel svm on xor data generated from gaussians
'''
def run_xor_standard(visualize=False):
    k,kparam,reg,N,noise,X,y,iterations = get_settings()
    w,errors = svm_kernel.fit_svm_kernel(X, y, its=iterations, kernel=(k, (kparam)), C=reg, visualize=visualize)
    #svm_kernel_new.make_plot_twoclass(X,y,w,kernel=(k,(kparam)))
    return errors
'''
runs double random kernel svm on xor data generated from gaussians
'''
def run_xor_drandom(visualize=False):
    k,kparam,reg,N,noise,X,y,iterations = get_settings()
    w,errors = svm_kernel.fit_svm_kernel_double_random(X, y, its=iterations * N, kernel=(k, (kparam)), C=reg, visualize=visualize)
    #svm_kernel_new.make_plot_twoclass(X,y,w,kernel=(k,(kparam)))
    return errors


if __name__ == '__main__':
    plt.ion()
    visualize=False
    stds = []
    drns = []
    drts = []
    k,kparam,reg,N,noise,X,y,iterations = get_settings()
    X_test,Y_test = svm_kernel.make_data_xor(N, noise)
    for i in range(0,100):

        # generate test data:
        # print i,"standard"
        w,errors = svm_kernel.fit_svm_kernel(X, y, its=iterations, kernel=(k, (kparam)), C=reg, visualize=visualize)
        save_results("./res/experiments_iterative_random/standard_" + str(i) + ".res",errors)
        test_error_std = svm_kernel.test_svm(X_test, Y_test, w, (k,(kparam)))[0]
        # print "error on test set:",test_error_std

        # print i,"double random"
        w,errors = svm_kernel.fit_svm_kernel_double_random(X, y, its=iterations * N, kernel=(k, (kparam)), C=reg, visualize=visualize)
        save_results("./res/experiments_iterative_random/drandom_" + str(i) + ".res",errors)
        test_error_dr = svm_kernel.test_svm(X_test, Y_test, w, (k,(kparam)))[0]
        # print "error on test set:",test_error_dr

        # print i,"double random threading"
        w,errors = fit_svm_kernel_double_random_threading(k, kparam, reg, N, noise, X, y, iterations, visualize=visualize)
        test_error_drt = svm_kernel.test_svm(X_test, Y_test, w, (k,(kparam)))[0]
        save_results("./res/experiments_iterative_random/drandom_threading_" + str(i) + ".res",errors)
        # print "error on test set:",test_error_drt

        print "comparison:",i
        print "std:\t","drn:\t","drt:\t"
        # print("%.2f\t%.2f\t%.2f" % (test_error_std,test_error_dr,test_error_drt))
        stds.append(test_error_std)
        drns.append(test_error_dr)
        drts.append(test_error_drt)

        print "intermediates:"
        print "comparison:",i

        print "mean:",np.mean(stds),"std:",np.std(stds)
        print "mean:",np.mean(drns),"std:",np.std(drns)
        print "mean:",np.mean(drts),"std:",np.std(drts)
    print "finals:"
    print "comparison:",i

    print "mean:",np.mean(stds),"std:",np.std(stds)
    print "mean:",np.mean(drns),"std:",np.std(drns)
    print "mean:",np.mean(drts),"std:",np.std(drts)

# if __name__ == '__main__':
#     plt.ion()
#     for i in range(0,100):
#         print i,"standard"
#         errors = run_xor_standard(visualize=False)
#         save_results("./res/experiments_iterative_random/standard_" + str(i) + ".res",errors)
#         print i,"double random"
#         errors = run_xor_drandom(visualize=True)
#         save_results("./res/experiments_iterative_random/drandom_" + str(i) + ".res",errors)
#         print i,"double random threading"
