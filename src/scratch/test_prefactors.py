import numpy as np
from fractions import Fraction

# Physical units/constants
sigma_sb = 5.670374419e-5
clight = 2.99792458e10
arad = 4.0 * sigma_sb / clight
KELVIN_PER_HEV = 1.160451812e6

# Base parameters
alpha = Fraction(3, 2)
beta = Fraction(8, 5)
lambdap = Fraction(1, 5)
mu = Fraction(7, 50)
gamma = Fraction(5, 4)
r = gamma - 1
omega = Fraction(0)
rho0 = Fraction(1932, 100)

n = (4 + alpha) / beta
k = 1 - mu
q = k * n + lambdap - 1

# Denominator of exponents
denom = 4 + 2*lambdap - 4*mu

# Exponents for tau = 0
tau = Fraction(0)
a = (2*beta - beta*lambdap - 8 - 2*alpha + mu*(4+alpha)) / denom
b = (mu - 2) / denom
c = (3*mu - 2*lambdap - 2 + tau*(2*(beta-alpha-4) - beta*lambdap + mu*(4+alpha))) / denom

a1 = -2 * (4+alpha-2*beta) / denom
b1 = -2 / denom
c1 = -2 * (tau*(4+alpha-2*beta) - 1) / denom

a2 = a1 - a
b2 = b1 - b
c2 = c1 - c - 1

a3 = a1 - 2*a
b3 = b1 - 2*b
c3 = c1 - 2*c - 2

print(f"n = {n} = {float(n)}")
print(f"k = {k} = {float(k)}")
print(f"q = {q} = {float(q)}")
print(f"denom = {denom} = {float(denom)}")
print(f"a = {a} = {float(a)}")
print(f"b = {b} = {float(b)}")
print(f"c = {c} = {float(c)}")
print(f"a1 = {a1} = {float(a1)}")
print(f"b1 = {b1} = {float(b1)}")
print(f"c1 = {c1} = {float(c1)}")
print(f"a2 = {a2} = {float(a2)}")
print(f"b2 = {b2} = {float(b2)}")
print(f"c2 = {c2} = {float(c2)}")
print(f"a3 = {a3} = {float(a3)}")
print(f"b3 = {b3} = {float(b3)}")
print(f"c3 = {c3} = {float(c3)}")
