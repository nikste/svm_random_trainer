import random

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

#test_plot()
#test_plot_two_graphs()