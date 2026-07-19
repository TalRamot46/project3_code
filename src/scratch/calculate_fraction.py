from sympy import Rational, symbols, solve, simplify

# Let's define the parameters as exact rational numbers
alpha = Rational(3, 2)     # 1.5
beta = Rational(8, 5)      # 1.6
lambdap = Rational(1, 5)   # 0.2
mu = Rational(7, 50)       # 0.14
r = Rational(1, 4)         # 0.25
gamma = r + 1              # 1.25

# n = (4 + alpha) / beta
n = (4 + alpha) / beta
# k = 1 - mu
k = 1 - mu
# q = k * n + lambdap - 1
q = k * n + lambdap - 1

# denom = 1 / (4 + 2 * lambdap - 4 * mu)
denom = 1 / (4 + 2 * lambdap - 4 * mu)

# c = denom * (3*mu - 2*lambdap - 2 + tau*(2*(beta - alpha - 4) - beta*lambdap + mu*(4 + alpha)))
# Let's write c as c_const + tau * c_tau
c_const = denom * (3 * mu - 2 * lambdap - 2)
c_tau = denom * (2 * (beta - alpha - 4) - beta * lambdap + mu * (4 + alpha))

# c1 = -2 * denom * (tau * (4 + alpha - 2 * beta) - 1)
# c1 = c1_const + tau * c1_tau
c1_const = -2 * denom * (-1)
c1_tau = -2 * denom * (4 + alpha - 2 * beta)

# c3 = -2 * denom * (tau * (mu * (4 + alpha) - beta * lambdap - 4 - alpha) + 1 - mu)
# c3 = c3_const + tau * c3_tau
c3_const = -2 * denom * (1 - mu)
c3_tau = -2 * denom * (mu * (4 + alpha) - beta * lambdap - 4 - alpha)

# cS = c + (q + 1) * c1 + n * c3
# cS = cS_const + tau * cS_tau
cS_const = c_const + (q + 1) * c1_const + n * c3_const
cS_tau = c_tau + (q + 1) * c1_tau + n * c3_tau

print(f"n: {n}")
print(f"k: {k}")
print(f"q: {q}")
print(f"denom: {denom}")
print(f"c_const: {c_const}")
print(f"c_tau: {c_tau}")
print(f"c1_const: {c1_const}")
print(f"c1_tau: {c1_tau}")
print(f"c3_const: {c3_const}")
print(f"c3_tau: {c3_tau}")
print(f"cS_const: {cS_const}")
print(f"cS_tau: {cS_tau}")

# We want cS = 0 for constant flux, so tau_const_flux = -cS_const / cS_tau
tau_const_flux = -cS_const / cS_tau
print(f"Exact tau for constant flux: {tau_const_flux}")
print(f"Approx tau for constant flux: {float(tau_const_flux)}")

# For this tau, what is cS? It is 0 by definition.
# What is c?
c_val = c_const + tau_const_flux * c_tau
print(f"c for constant flux: {c_val} (approx {float(c_val)})")

# What is self.bath_power = cS - 4 * tau?
# Since cS = 0, bath_power = -4 * tau
bath_power = -4 * tau_const_flux
print(f"Exact bath_power: {bath_power}")
print(f"Approx bath_power: {float(bath_power)}")
