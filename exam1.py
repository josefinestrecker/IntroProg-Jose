import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fsolve

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
sigma_values = [1.001, 1.5]
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
    

# Define the utility function
def utility(L, G, alpha, v):
    return np.log(L**alpha * G**(1 - alpha)) - v * L**2 / 2

# Question 4: Find the socially optimal tax rate tau* maximizing worker utility
def negative_utility(tau):
    w_tilde = (1 - tau) * w
    L = optimal_labor_supply(w_tilde, kappa, alpha, v)
    return -utility(L, G, alpha, v)

# Define the new utility function
def utility_1(L, G, alpha, sigma, rho, v, epsilon):
    C = kappa + (1 + tau) * w * L
    return ((alpha * C**((sigma - 1) / sigma) + (1 - alpha) * G**((sigma - 1) / sigma))**((1 - rho) / sigma) - 1) / (1 - rho) - v * L**(1 + epsilon) / (1 + epsilon)

# Define the function for the equilibrium condition
def equilibrium(G, w_tilde, alpha, sigma, rho, v, epsilon):
    L = optimal_labor_supply(w_tilde, kappa, alpha, v)
    return G - tau * w * L

def negative_utility_with_equilibrium(tau, w, alpha, sigma, rho, v, epsilon):
    w_tilde = (1 - tau) * w
    G = fsolve(equilibrium, 1.0, args=(w_tilde, alpha, sigma, rho, v, epsilon))
    L = optimal_labor_supply(w_tilde, kappa, alpha, v)
    return -utility_2(L, G, alpha, sigma, rho, v, epsilon)


#Problem 2
# Set up the parameters
eta_2 = 0.5
w_2 = 1.0
kappa_values_2 = [1.0, 2.0]
rho_2 = 0.9
iota_2 = 0.01
sigma_epsilon_2 = 0.1
R_2 = (1 + 0.01)**(1/12)
K_2 = 10000  # Number of random shock series
T_2 = 120  # Number of periods


# Define the function for optimal labor supply choice
def optimal_labor_supply(kappa_2, eta_2, w_2):
    return ((1 - eta_2) * kappa_2 / w_2)**(1 / eta_2)

# Define the function for profits
def profits(kappa_2, l_2, eta_2, w_2):
    return kappa_2 * l_2**(1 - eta_2) - w_2 * l_2