import numpy as np
import matplotlib.pyplot as plt


def normal_dist(x, mean, sd):
    """calculates normal distribution"""
    prob_density = (np.pi*sd) * np.exp(-0.5 * ((x-mean)/sd)**2)
    plt.plot(x,prob_density, color = 'red')
    plt.xlabel('Data Points')
    plt.ylabel('Probability Density')






def marginal_dist():
    """calculates mariginal distribution """

    return 0






