import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fsolve
from scipy.stats import norm

#Problem 1

# Set up the parameters
alpha = 0.5
kappa = 1.0
v = 1/(2 * 16**2)
w = 1.0
tau = 0.3
G_values = [1.0, 2.0]
w_values = np.linspace(0.1, 2.0, 100)  # Create a range of w values
tau_values = np.linspace(0.1, 0.9, 100)  # Create a range of tau values
sigma_values = [0.001, 1.5]
rho_values = [1.001, 1.5]
epsilon = 1.0

# Calculate w_tilde
w_tilde = (1 - tau) * w

# Define the function for optimal labor supply choice
def optimal_labor_supply(w_tilde, kappa, alpha, v):
    return (np.sqrt(kappa**2 + 4 * alpha/v * w_tilde**2) - kappa) / (2 * w_tilde)

# Calculate optimal labor supply for each G
for G in G_values:
    L_star = optimal_labor_supply(w_tilde, kappa, alpha, v)
    print(f"For G = {G}, the optimal labor supply choice is {L_star}")

# Define the utility function
def utility(L, G, alpha, v):
    return np.log(L**alpha * G**(1 - alpha)) - v * L**2 / 2

# Question 4: Find the socially optimal tax rate tau* maximizing worker utility
def negative_utility(tau):
    w_tilde = (1 - tau) * w
    L = optimal_labor_supply(w_tilde, kappa, alpha, v)
    return -utility(L, G, alpha, v)


