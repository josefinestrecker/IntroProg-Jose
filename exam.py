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
sigma_values = [1.001, 1.5]
rho_values = [1.001, 1.5]
epsilon = 1.0

#Define the utility function
def utility(L, G, alpha, v):
    return np.log(L**alpha * G**(1 - alpha)) - v * L**2 / 2

# Calculate w_tilde
w_tilde = (1 - tau) * w

# Define the function for optimal labor supply choice
def optimal_labor_supply(w_tilde, kappa, alpha, v):
    return (np.sqrt(kappa**2 + 4 * alpha/v * w_tilde**2) - kappa) / (2 * w_tilde)

# Calculate optimal labor supply for each G
for G in G_values:
    L_star = optimal_labor_supply(w_tilde, kappa, alpha, v)
    print(f"For G = {G}, the optimal labor supply choice is {L_star}")

# Question 2: Illustrate how L*(w_tilde) depends on w
L_star_values = [optimal_labor_supply((1 - tau) * w, kappa, alpha, v) for w in w_values]
plt.figure(figsize=(10, 6))
plt.plot(w_values, L_star_values)
plt.xlabel('w')
plt.ylabel('L*(w_tilde)')
plt.title('Dependence of Optimal Labor Supply on w')
plt.grid(True)
plt.show()

# Question 3: Plot the implied L, G, and worker utility for a grid of tau-values
#Defining the new G
G_values = [tau * w * L_star*((1-tau)*w) for tau, L_star in zip(tau_values, L_star_values)]  # Calculate G for each tau
utilities = [utility(L, G, alpha, v) for L, G in zip(L_star_values, G_values)]
plt.figure(figsize=(10, 6))
plt.plot(tau_values, L_star_values, label='L')
plt.plot(tau_values, G_values, label='G')
plt.plot(tau_values, utilities, label='Utility')
plt.xlabel('tau')
plt.ylabel('Value')
plt.title('Implied L, G, and Worker Utility for Different Tau Values')
plt.legend()
plt.grid(True)
plt.show()

# Question 4: Find the socially optimal tax rate tau* maximizing worker utility
def negative_utility(tau):
    w_tilde = (1 - tau) * w
    L = optimal_labor_supply(w_tilde, kappa, alpha, v)
    return -utility(L, G, alpha, v)

result = minimize(negative_utility, 0.5, bounds=[(0.1, 0.9)])
optimal_tau = result.x[0]
print(f"The socially optimal tax rate is {optimal_tau}")

# Plot the utility as a function of tau
utilities = [utility(optimal_labor_supply((1 - tau) * w, kappa, alpha, v), G, alpha, v) for tau in tau_values]
plt.figure(figsize=(10, 6))
plt.plot(tau_values, utilities)
plt.xlabel('tau')
plt.ylabel('Utility')
plt.title('Worker Utility for Different Tau Values')
plt.axvline(x=optimal_tau, color='r', linestyle='--', label=f'Optimal tau = {optimal_tau}')
plt.legend()
plt.grid(True)
plt.show

#This code extends the previous code by adding plots 
# for the dependence of optimal labor supply on w and the implied 
# L, G, and worker utility for a grid of tau values. It also uses 
# the scipy.optimize.minimize function to find the socially optimal 
# tax rate that maximizes worker utility. 
# The negative_utility function is defined because minimize 
# finds the minimum of a function, and we want to find the maximum utility.

# Define the new utility function
def utility(L, G, alpha, sigma, rho, v, epsilon):
    C = kappa + (1 + tau) * w * L
    return (((alpha * C**((sigma - 1) / sigma) + (1 - alpha) * G**((sigma-1) / (sigma)))**((sigma / (sigma-1)))**((1 - rho)) - 1) / (1 - rho)) - v * L**(1 + epsilon) / (1 + epsilon)

# Define the function for the equilibrium condition
def equilibrium(G, w_tilde, alpha, sigma, rho, v, epsilon):
    L = optimal_labor_supply(w_tilde, kappa, alpha, v)
    return G - tau * w_tilde * L

# Question 5: Find the G that solves the equilibrium condition
for sigma, rho in zip(sigma_values, rho_values):
    G_solution = fsolve(equilibrium, 1.0, args=(w_tilde, alpha, sigma, rho, v, epsilon))
    print(f"For sigma = {sigma} and rho = {rho}, the solution for G is {G_solution[0]}")

# Question 6: Find the socially optimal tax rate tau* maximizing worker utility while keeping the equilibrium condition
def negative_utility_with_equilibrium(tau, w, alpha, sigma, rho, v, epsilon):
    w_tilde = (1 - tau) * w
    G = fsolve(equilibrium, 1.0, args=(w_tilde, alpha, sigma, rho, v, epsilon))
    L = optimal_labor_supply(w_tilde, kappa, alpha, v)
    return -utility(L, G, alpha, sigma, rho, v, epsilon)

for sigma, rho in zip(sigma_values, rho_values):
    result = minimize(negative_utility_with_equilibrium, 0.5, args=(w, alpha, sigma, rho, v, epsilon), bounds=[(0.1, 0.9)])
    optimal_tau = result.x[0]
    print(f"For sigma = {sigma} and rho = {rho}, the socially optimal tax rate is {optimal_tau}")


#This code extends the previous code by adding the 
# new utility function and the equilibrium condition. 
# It then uses scipy.optimize.fsolve to find the G that solves 
# the equilibrium condition for each set of sigma and rho values. 
# Finally, it finds the socially optimal tax rate tau* that maximizes
#  worker utility while keeping the equilibrium condition, again for each 
# set of sigma and rho values.




#Problem 2
import numpy as np
from scipy.stats import norm

# Set up the parameters
eta = 0.5
w = 1.0
kappa_values = [1.0, 2.0]
rho = 0.9
iota = 0.01
sigma_epsilon = 0.1
R = (1 + 0.01)**(1/12)
K = 10000  # Number of random shock series
T = 120  # Number of periods

# Define the function for optimal labor supply choice
def optimal_labor_supply(kappa, eta, w):
    return ((1 - eta) * kappa / w)**(1 / eta)

# Define the function for profits
def profits(kappa, l, eta, w):
    return kappa * l**(1 - eta) - w * l

# Question 1: Verify numerically that the optimal labor supply choice maximizes profits
for kappa in kappa_values:
    l_star = optimal_labor_supply(kappa, eta, w)
    profit = profits(kappa, l_star, eta, w)
    print(f"For kappa = {kappa}, the optimal labor supply choice is {l_star} and the profit is {profit}")

# Generate random shock series
np.random.seed(0)  # For reproducibility
epsilon_series = np.random.normal(loc=-0.5*sigma_epsilon**2, scale=sigma_epsilon, size=(K, T))
kappa_series = np.exp(rho * np.log(np.append(np.ones((K, 1)), np.exp(epsilon_series[:, :-1]), axis=1)) + epsilon_series)

# Question 2: Calculate the ex ante expected value of the salon
H_values = []
for k in range(K):
    kappa_k = kappa_series[k, :]
    l_prev = optimal_labor_supply(kappa_k[0], eta, w)
    total_profit = profits(kappa_k[0], l_prev, eta, w)
    for t in range(1, T):
        l = optimal_labor_supply(kappa_k[t], eta, w)
        total_profit += R**(-t) * (profits(kappa_k[t], l, eta, w) - (l != l_prev) * iota)
        l_prev = l
    H_values.append(total_profit)
H = np.mean(H_values)
print(f"The ex ante expected value of the salon is {H}")

# Define the function for the policy
def policy(l_prev, l_star, Delta):
    return l_star if abs(l_prev - l_star) > Delta * abs(l_prev) else l_prev

# Question 3: Calculate H if the policy above was followed with Delta=0.005
Delta = 0.005
H_values = []
for k in range(K):
    kappa_k = kappa_series[k, :]
    l_prev = optimal_labor_supply(kappa_k[0], eta, w)
    total_profit = profits(kappa_k[0], l_prev, eta, w)
    for t in range(1, T):
        l_star = optimal_labor_supply(kappa_k[t], eta, w)
        l = policy(l_prev, l_star, Delta)
        total_profit += R**(-t) * (profits(kappa_k[t], l, eta, w) - (l != l_prev) * iota)
        l_prev = l
    H_values.append(total_profit)
H = np.mean(H_values)
print(f"The ex ante expected value of the salon with Delta = {Delta} is {H}")

# Question 4: Find the optimal Delta maximizing H
def negative_H(Delta):
    H_values = []
    for k in range(K):
        kappa_k = kappa_series[k, :]
        l_prev = optimal_labor_supply(kappa_k[0], eta, w)
        total_profit = profits(kappa_k[0], l_prev, eta, w)
        for t in range(1, T):
            l_star = optimal_labor_supply(kappa_k[t], eta, w)
            l = policy(l_prev, l_star, Delta)
            total_profit += R**(-t) * (profits(kappa_k[t], l, eta, w) - (l != l_prev) * iota)
            l_prev = l
        H_values.append(total_profit)
    return -np.mean(H_values)

result = minimize(negative_H, 0.005, bounds=[(0.001, 0.01)])
optimal_Delta = result.x[0]
print(f"The optimal Delta is {optimal_Delta}")

# Question 5: Suggest an alternative policy you believe might improve profitability. Implement and test your policy.
# Here, I suggest a policy that hires or fires only if the difference between l_prev and l_star is greater than Delta * l_star instead of Delta * l_prev.
def alternative_policy(l_prev, l_star, Delta):
    return l_star if abs(l_prev - l_star) > Delta * abs(l_star) else l_prev

H_values = []
for k in range(K):
    kappa_k = kappa_series[k, :]
    l_prev = optimal_labor_supply(kappa_k[0], eta, w)
    total_profit = profits(kappa_k[0], l_prev, eta, w)
    for t in range(1, T):
        l_star = optimal_labor_supply(kappa_k[t], eta, w)
        l = alternative_policy(l_prev, l_star, optimal_Delta)
        total_profit += R**(-t) * (profits(kappa_k[t], l, eta, w) - (l != l_prev) * iota)
        l_prev = l
    H_values.append(total_profit)
H = np.mean(H_values)
print(f"The ex ante expected value of the salon with the alternative policy is {H}")



#Problem 3

import numpy as np
from scipy.optimize import minimize

# Define the Griewank function
def griewank(x): 
    A = np.sum(x**2 / 4000)
    B = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return A - B + 1

# Bounds for x and tolerance τ > 0.
bounds = [(-600, 600), (-600, 600)]
τ = 1e-8

# The number of warm-up iterations, K > 0, and the maximum number of iterations, underlined_K > K.
K = 1000
underlined_K = 10

# Optimal solution
optimal_solution = None
optimal_value = np.inf

# Store effective initial guesses
initial_guesses = []

# Iterations
for k in range(K):
    # A. Draw a random x^k uniformly within the chosen bounds.
    x0 = np.random.uniform(bounds[0][0], bounds[0][1], 2)
    initial_guesses.append(x0)
    
    # B. If k < underlined_K, go to step E.
    if k < underlined_K:     #We skip to E
        # E. Run the optimizer with x^k0 as initial guess and get the result x^(k*)
        res = minimize(griewank, x0, method='BFGS', tol=τ)
    
        # F. Set x^* = x^(k*) if k = 0 or f(x^(k*)) < f(x^*)
        if res.fun < optimal_value:
            optimal_value = res.fun
            optimal_solution = res.x
            
        # G. If f(x^*) < τ, go to step 4.
        if optimal_value < τ:
            break
# 4. Return the result x^*.
print("Optimal solution:", optimal_solution)
print("Optimal value:", optimal_value)

# Show initial guesses and how they vary
for i, guess in enumerate(initial_guesses):
    print("Iteration {}, initial guess: {}".format(i, guess))


#in 3:
#We iterate underlined_K times.
#In each iteration, we first draw a random vector x0 uniformly within the chosen bounds.
#If the iteration count k is less than K, we proceed with the optimizer and run it with x0 as the initial guess (Step E). The result x^(k*) is obtained from the optimizer.
#Then, we set x^* to be equal to x^(k*) if it's the first iteration (i.e., k=0) or if the function value of the new result f(x^(k*)) is less than the function value of our current best solution f(x^*) (Step F).
#If k is not less than K, we skip the optimization step (Steps E and F) and continue with the next iteration.
#We repeat these steps until we have completed underlined_K iterations.
#Further we print out the optimal solution and the optimal value.
#Finally, we use the funciton break, which will lead us directly to 4, if f(x^*) < tau

#the optimal solution found suggests that when $x_1$ is approximately -50.24035172 and $x_2$ is approximately 35.50754101, the Griewank function is at its minimum within the defined bounds and given the initial guesses and tolerances.

#4.2
#to snwer the question: "Is it a better idea to set ▁K=100? Is the convergense faster?"
# we should just change the value of underlined_K to 100 from 10 and analyze it. 
#however increasing the iterations of underlined_K can potentially lead to better solutions, as we are sampling more initial points, 
# increasing the chance of starting near the global minimu
# But this does not necessarily mean that the convergence will be faster. 
# In fact, increasing underlined_K will likely increase the total runtime of your program, aswe are performing more optimization runs.


