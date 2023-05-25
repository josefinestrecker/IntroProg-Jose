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
L_values = [optimal_labor_supply((1 - tau) * w, kappa, alpha, v) for tau in tau_values]
utilities = [utility(L, G, alpha, v) for L, G in zip(L_values, G_values)]
plt.figure(figsize=(10, 6))
plt.plot(tau_values, L_values, label='L')
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


##CHAT BESKRIVER KODERNE FRA 2-4 SÅDAN HER
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
    return ((alpha * C**((sigma - 1) / sigma) + (1 - alpha) * G**((sigma - 1) / sigma))**((1 - rho) / sigma) - 1) / (1 - rho) - v * L**(1 + epsilon) / (1 + epsilon)

# Define the function for the equilibrium condition
def equilibrium(G, w_tilde, alpha, sigma, rho, v, epsilon):
    L = optimal_labor_supply(w_tilde, kappa, alpha, v)
    return G - tau * w * L

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


##CHAT BESKRIVER KODERNE FRA 5-6 SÅDAN HER
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


#IGEN HHER ER DER BESRKIVELSE AF KODERNE FOR DE TO FØRSTE SPØRGSMÅL

#This code first sets up the parameters as given in the problem. 
# It then defines a function optimal_labor_supply that calculates the optimal 
# labor supply given kappa, eta, and w, and a function profits that calculates 
# the profits given kappa, l, eta, and w. It verifies numerically that the 
# optimal labor supply choice maximizes profits for each kappa in kappa_values.

#Question 2, it generates K random shock series epsilon_series and calculates 
# the corresponding kappa_series. It then calculates the ex ante expected value 
# of the salon H by summing the discounted profits for each period and each 
# shock series, taking into account the adjustment cost iota if the labor 
# supply choice changes from the previous period. The expected value is 
# then approximated by the mean of these total profits over all shock series.


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

#IGEN CHAT BESKRIVELSER
#Question 4: we are finding the optimal Delta value that maximizes the 
# value of H. The function negative_H(Delta) calculates the average value 
# of H for a given Delta value. It iterates through the shock series and 
# calculates the total profit by following the given policy with the Delta value.
#  The negative sign is used because we are using the minimize function to find 
# the maximum of H. The minimize function from the SciPy library is used to 
# find the minimum of the negative_H function, which essentially finds the maximum 
# of H. The result contains the optimal Delta value, which is then printed.

#In Question 5, an alternative policy is suggested to improve profitability. 
# The function alternative_policy(l_prev, l_star, Delta) is defined, which 
# determines whether to hire or fire hairdressers based on the difference between 
# the previous labor supply l_prev and the optimal labor supply l_star. 
# If the difference is greater than Delta times the absolute value of l_star, 
# a hiring or firing decision is made; otherwise, the previous labor supply is 
# maintained. The code then calculates the value of H using this alternative policy by 
# following a similar process as in Question 4.#