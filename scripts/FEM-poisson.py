import numpy as np
from scipy import integrate

# simple problem: Find u, such that -u'' = 1 on (0,1)
# with u(0) = u(1) = 0

# domain discretization
a = 0
b = 1
N = 30 # number of nodes, N-2 is the number of free nodes
h = (b-a)/(N-1)
A = np.zeros([N,N])
f = np.zeros(N)
u = np.zeros(N)

# compute matrix and vector components (integrals done manually)
Aii = 2/h
Aij = -1/h
fi = h

for i in range(N-1):
    A[i,i] = Aii
    A[i,i+1] = Aij
    A[i+1,i] = Aij
    f[i] = fi
A[N-1,N-1] = Aii
A = A[1:N-1,1:N-1]
f = f[1:N-1]

# solve system Au=f
u[1:N-1] = np.linalg.solve(A,f)
# print(u)

# error in energy norm, ||u||^2 := a(u,u)
# galerkin orthog. => ||u-uh||^2 = ||u||^2 - ||uh||^2
norm_uh = 0
for i in range(len(u)): norm_uh += u[i]*h

u = lambda x: -0.5*(-1+x)*x
norm_u = integrate.quad(u, 0, 1)[0]

print("Error in energy norm =",(abs(norm_uh - norm_u))**0.5)