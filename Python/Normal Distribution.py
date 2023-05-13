# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:46:21 2023

@author: Gbenga Agunbiade
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def simulate_experiment(true_mean, true_std, num_samples):
    """
    Simulate an experiment with a normal distribution.
    Returns a sample of random values from the distribution.
    """
    samples = np.random.normal(true_mean, true_std, size=num_samples)
    return samples

def likelihood(mean, std, samples):
    """
    Compute the likelihood of the mean given the observed data (samples).
    """
    likelihood = np.prod(norm.pdf(samples, loc=mean, scale=std))
    return likelihood

def estimate_parameter(samples):
    """
    Estimate the parameter (mean) using maximum likelihood estimation.
    Returns the estimated parameter value.
    """
    return np.mean(samples)

# Parameters
true_mean = 10.0
true_std = 2.4
num_samples = 200
num_experiments = 5000

# Simulate experiments and estimate the parameter
estimates = []
for _ in range(num_experiments):
    samples = simulate_experiment(true_mean, true_std, num_samples)
    parameter_estimate = estimate_parameter(samples)
    estimates.append(parameter_estimate)

# Calculate confidence interval
confidence_level = 0.95
lower_bound = np.percentile(estimates, (1 - confidence_level) / 2 * 100)
upper_bound = np.percentile(estimates, (1 + confidence_level) / 2 * 100)

# Plot histogram of parameter estimates
plt.hist(estimates, bins=30)
plt.axvline(lower_bound, color='b', linestyle='--', label='Confidence Interval')
plt.axvline(upper_bound, color='b', linestyle='--')
plt.xlabel('Mean (Parameter Estimate)')
plt.ylabel('Frequency')
plt.title('Histogram of Parameter Estimates')
plt.legend()
plt.show()

# Plot density function with confidence interval
x = np.linspace(true_mean - 4 * true_std, true_mean + 4 * true_std, 5000)
y = norm.pdf(x, loc=true_mean, scale=true_std)
plt.plot(x, y, label='True Distribution')
plt.fill_between(x, y, where=(x >= lower_bound) & (x <= upper_bound), color='red', alpha=0.6, label='Confidence Interval')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('True Distribution with Confidence Interval')
plt.legend()
plt.show()
