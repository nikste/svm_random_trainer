import pylab as pl
import scipy as sp
from scipy.linalg import eig
from numpy.random import multivariate_normal as mvn
import pdb
from scipy.spatial.distance import cdist

def GaussianKernel(X1, X2, sigma):
   assert(X1.shape[0] == X2.shape[0])
   K = cdist(X1.T, X2.T, 'euclidean')
   K = sp.exp(-(K ** 2) / (2. * sigma ** 2))
   return K

def linearKernel(X1, X2, sigma = 1):
    assert(X.shape[0] == X2.shape[0])
    K = cdist(X1, X2, 'euclidean')
    return K


def fit_svm_kernel(X,Y,its=100,eta=1.,C=.1,kernel=(GaussianKernel,(1.))):
	D,N = X.shape[0],X.shape[1]
	X = sp.vstack((sp.ones((1,N)),X))
	W = sp.randn(N)
	for it in range(its):
		rn = sp.random.randint(N)
		yhat = predict_svm_kernel(X[:,rn],X,W,kernel)
		if yhat*Y[:,rn] > 1: G = C * W
		else: G = C * W - Y[:,rn] * kernel[0](sp.vstack((X[:,rn] )),X,kernel[1]).flatten()
		W -= eta/(it+1.) * G
		print test_svm(X,Y,W,(k,(kparam)))
		if it % 100 == 0:
			make_plot_twoclass(X[1:3,:],Y,W,kernel=kernel)
	return W

def fit_svm_kernel_double_random(X,Y,its=100,eta=1.,C=.1,kernel=(GaussianKernel,(1.))):
    D,N = X.shape[0],X.shape[1]
    X = sp.vstack((sp.ones((1,N)),X))
    W = sp.randn(N)
    for it in range(its):
		rn = sp.random.randint(N)
		rn2 = sp.random.randint(N)
		yhat = predict_svm_kernel_double_random(X[:,rn],X[:,rn2],W[rn2],kernel)
		if yhat*Y[:,rn] > 1:
			G = C * W[rn2]
		else:
			G = C * W[rn2] - Y[:,rn] * kernel[0](sp.vstack((X[:,rn] )),sp.vstack((X[:,rn2])),kernel[1]).flatten()
		W[rn2] -= eta/(it+1.) * G
		print test_svm(X,Y,W,(k,(kparam)))
		if it % 100 == 0:
			make_plot_twoclass(X[1:3,:],Y,W,kernel=kernel)
    return W


def predict_svm_kernel(x,xt,w,kernel):
	return w.dot(kernel[0](sp.vstack((x)),xt,kernel[1]).T)

def predict_svm_kernel_double_random(x,xt,w,kernel):
    return (w * kernel[0](sp.vstack((x)),sp.vstack((xt)),kernel[1]).T)[0]

def make_data_twoclass(N=50):
	# generates some toy data
	mu = sp.array([[0,2],[0,-2]]).T
	C = sp.array([[5.,4.],[4.,5.]])
	X = sp.hstack((mvn(mu[:,0],C,N/2).T, mvn(mu[:,1],C,N/2).T))
	Y = sp.hstack((sp.ones((1,N/2.)),-sp.ones((1,N/2.))))
	return X,Y

def make_data_xor(N=80,noise=.25):
	# generates some toy data
	mu = sp.array([[-1,1],[1,1]]).T
	C = sp.eye(2)*noise
	X = sp.hstack((mvn(mu[:,0],C,N/4).T,mvn(-mu[:,0],C,N/4).T, mvn(mu[:,1],C,N/4).T,mvn(-mu[:,1],C,N/4).T))
	Y = sp.hstack((sp.ones((1,N/2.)),-sp.ones((1,N/2.))))
	return X,Y

def make_data_cos(N=100,noise=.3):
	# generates some toy data
	x = sp.randn(1,N)*sp.pi
	y = sp.cos(x) + sp.randn(1,N) * noise
	return x,y

def make_plot_twoclass(X,Y,W,kernel):
	fig = pl.figure(figsize=(5,4))
	fig.clf()
	colors = "brymcwg"

	# Plot the decision boundary.
	h = .2 # stepsize in mesh
	x_min, x_max = X[0,:].min() - 1, X[0,:].max() + 1
	y_min, y_max = X[1,:].min() - 1, X[1,:].max() + 1
	xx, yy = sp.meshgrid(sp.arange(x_min, x_max, h),
                     sp.arange(y_min, y_max, h))

	Z = predict_svm_kernel(sp.c_[sp.ones(xx.ravel().shape[-1]), xx.ravel(), yy.ravel()].T,sp.vstack((sp.ones((1,X.shape[-1])),X)),W,kernel).reshape(xx.shape)
	cs = pl.contourf(xx, yy, Z,alpha=.5)
	pl.axis('tight')
	pl.colorbar()
	pl.axis('equal')
	y = sp.maximum(0,-Y)+1
	# plot the data
	pl.hold(True)

	ypred = 	W.T.dot(kernel[0](X,X,kernel[1]).T)
	for ic in sp.unique(y):
		idx = (y == int(ic)).flatten()
		sv = (Y.flatten()[idx]*ypred[idx] < 1)
		pl.plot(X[0,idx.nonzero()[0][sv]], X[1,idx.nonzero()[0][sv]], colors[int(ic)]+'o',markersize=13)
		pl.plot(X[0,idx.nonzero()[0][~sv]], X[1,idx.nonzero()[0][~sv]], colors[int(ic)]+'o',markersize=7)
	pl.axis('tight')

	pl.xlabel('$X_1$')
	pl.ylabel('$X_2$')

	#pl.title('SVM, Accuracy=%0.2f'%(Y==sp.sign(ypred)).mean())

	pl.show()
	# pl.savefig('../graphics/svm_kernel.pdf')

	fig = pl.figure(figsize=(5,5))
	fig.clf()
	colors = "brymcwg"
	for ic in sp.unique(y):
		idx = (y == int(ic)).flatten()
		pl.plot(X[0,idx], X[1,idx], colors[int(ic)]+'o',markersize=8)
	pl.axis('tight')

	# pl.xlabel('$X_1$')
	# pl.ylabel('$X_2$')
	# pl.xlim((x_min,x_max))
	# pl.ylim((y_min,y_max))
	# pl.grid()
	# pl.show()
	# pl.savefig('../graphics/svm_kernel_xor_data.pdf')

def test_svm(X,Y,W,(k,(kparam))):
	kernel = (k,(kparam))
	error = 0
	for rn in range(X.shape[1]):
		yhat = predict_svm_kernel(X[:,rn],X,W,kernel)
		if not yhat*Y[:,rn] >= 0:
			error = error + 1
	return error/float(X.shape[1])

if __name__ == '__main__':
	k = GaussianKernel
	kparam = 1.
	reg = .001
	N = 480
	noise = .1#.25
	X,y = make_data_xor(N,noise)
	iterations = 1000


	double_rand = True
	if double_rand:
		w = fit_svm_kernel_double_random(X,y,its=iterations,kernel=(k,(kparam)),C=reg)
		print "train error:",test_svm(X,y,w,(k,(kparam)))

		X_test,y_test = make_data_xor(N,noise)
		print "test error:",test_svm(X_test,y_test,w,(k,(kparam)))


		make_plot_twoclass(X_test,y_test,w,kernel=(k,(kparam)))
	else:
		w = fit_svm_kernel(X,y,its=iterations,kernel=(k,(kparam)),C=reg)
		print "train error:",test_svm(X,y,w,(k,(kparam)))

		X_test,y_test = make_data_xor(N,noise)
		print "test error:",test_svm(X_test,y_test,w,(k,(kparam)))


		make_plot_twoclass(X_test,y_test,w,kernel=(k,(kparam)))



