import random
import threading

import run_experiments
import matplotlib.pyplot as plt



def test_plot():
    plt.ion()
    data = []
    for i in range(0,10):
        rnd = random.randint(1,i+10)
        data.append(rnd)
        run_experiments.update_plot(data)

def test_plot_two_graphs():
    plt.ion()
    data = []
    for i in range(0,10):
        rnd1 = random.randint(0,i+10)
        rnd2 = random.randint(0,i+5)
        data.append([rnd1,rnd2])
        run_experiments.update_plot(data)


def plot_scatter(d1,d2):
	plt.plot(d1, d2, '-')
	plt.show()
	plt.pause(0.0000001) #Note this correction


plt.ion()
data = []
for i in range(0,100):
    rnd = random.randint(0,100)
    data.append([i*10,rnd])
    d1,d2 = zip(*data)
    plot_scatter(d1,d2)
#test_plot()
#test_plot_two_graphs()