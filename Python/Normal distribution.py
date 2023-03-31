# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 17:58:47 2023

@author: g361a609
"""

import numpy as np
from scipy.stats import norm

def simulate_experiment(mu, n_samples):
    """Simulate an experiment with a normal distribution and return the data"""
    return np.random.normal(mu, size=n_samples)

def likelihood(mu, data):
    """Calculate the likelihood function for a normal distribution"""
    return np.prod(norm.pdf(data, loc=mu))

def estimate_parameter(data):
    """Estimate the parameter mu using maximum likelihood estimation"""
    return np.mean(data)

# Simulate the experiment
true_mu = 5.0
n_samples = 3
data = simulate_experiment(true_mu, n_samples)

# Estimate the parameter using maximum likelihood estimation
mu_estimated = estimate_parameter(data)

print(f"True mu: {true_mu}")
print(f"Estimated mu: {mu_estimated}")

# Calculate the likelihood function for a range of mu values
mu_values = np.linspace(0, 10, 1000)
likelihoods = [likelihood(mu, data) for mu in mu_values]

# Plot the likelihood function
import matplotlib.pyplot as plt
plt.plot(mu_values, likelihoods)
plt.axvline(mu_estimated, color='red', linestyle='--')
plt.xlabel('mu')
plt.ylabel('Likelihood')
plt.title('Likelihood function for estimating mu')
plt.show()

n_bootstraps = 1000
bootstrapped_estimates = []
for i in range(n_bootstraps):
    bootstrap_data = np.random.choice(data, size=n_samples, replace=True)
    bootstrap_estimate = estimate_parameter(bootstrap_data)
    bootstrapped_estimates.append(bootstrap_estimate)

# Calculate the confidence interval
lower_bound = np.percentile(bootstrapped_estimates, 2.5)
upper_bound = np.percentile(bootstrapped_estimates, 97.5)
print(f"Estimated mu: {mu_estimated:.2f}")
print(f"Confidence interval: [{lower_bound:.2f}, {upper_bound:.2f}]")
