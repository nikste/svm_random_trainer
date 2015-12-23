import svm_kernel_new
import matplotlib.pyplot as plt



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
    k = svm_kernel_new.GaussianKernel
    kparam = 1.
    reg = .001
    iterations = 100

    N = 100
    noise = .1#.25
    X,y = svm_kernel_new.make_data_xor(N,noise)

    return [k,kparam,reg,N,noise,X,y,iterations]


def save_results(filename,data):
    f = open(filename,'w')
    for el in data:
        f.write(str(el)+"\n")
    f.close()

'''
runs standard kernel svm on xor data generated from gaussians
'''
def run_xor_standard(visualize=False):
    k,kparam,reg,N,noise,X,y,iterations = get_settings()
    w,errors = svm_kernel_new.fit_svm_kernel(X,y,its=iterations,kernel=(k,(kparam)),C=reg,visualize=visualize)
    #svm_kernel_new.make_plot_twoclass(X,y,w,kernel=(k,(kparam)))
    return errors
'''
runs double random kernel svm on xor data generated from gaussians
'''
def run_xor_drandom(visualize=False):
    k,kparam,reg,N,noise,X,y,iterations = get_settings()
    w,errors = svm_kernel_new.fit_svm_kernel_double_random(X,y,its=iterations*N,kernel=(k,(kparam)),C=reg,visualize=visualize)
    #svm_kernel_new.make_plot_twoclass(X,y,w,kernel=(k,(kparam)))
    return errors



if __name__ == '__main__':
    plt.ion()
    for i in range(0,100):
        print i,"standard"
        errors = run_xor_standard(visualize=False)
        save_results("./res/experiments_iterative_random/standard_" + str(i) + ".res",errors)
        print i,"double random"
        errors = run_xor_drandom()
        save_results("./res/experiments_iterative_random/drandom_" + str(i) + ".res",errors)
