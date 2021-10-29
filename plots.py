# -*- coding: utf-8 -*-
"""visualize the result."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(x_data,filepath,size1,size2):
    """visualize input data as histograms -> to help in pre-processing steps."""
    figure, axs = plt.subplots(size1,size2,figsize=(10,10))
    iterator = 0
    for i in range(size1):
        for j in range(size2):
            sns.histplot(x_data[:,iterator], ax = axs[i,j], bins = 23)
            axs[i,j].set_title("Feature " + str(iterator))
            axs[i,j].yaxis.set_visible(False)
            iterator+=1
    plt.tight_layout()
    plt.savefig(filepath)
    return str('save figure to ' + filepath)


def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")


def visualization(y, x, mean_x, std_x, w, save_name, is_LR=False):
    """visualize the raw data as well as the classification result."""
    fig = plt.figure()
    # plot raw data
    x = de_standardize(x, mean_x, std_x)
    #ax1 = fig.add_subplot(1, 2, 1)
    males = np.where(y == 1)
    females = np.where(y == -1)
    fig.scatter(x[males, 0], x[males, 1],marker='.', color=[0.06, 0.06, 1], s=20)
    fig.scatter(x[females, 0], x[females, 1], marker='*', color=[1, 0.06, 0.06], s=20)
    fig.set_xlabel("x=Height")
    fig.set_ylabel("y=Weight")
    fig.grid()
    


   