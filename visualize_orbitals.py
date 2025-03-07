import numpy as np
import matplotlib.pyplot as plt 
from pyscf import gto

distance = 0.735 #units in A
mol = gto.M(atom = [['H', (0,0,-distance/2)],['H', (0,0,distance/2)]], basis = 'sto-3g')

# one-electron hamiltonian
kin_ao = mol.intor("int1e_kin")
nuc_ao = mol.intor("int1e_nuc")
h1_ao = nuc_ao + kin_ao

# two-electron hamiltonian
h2_chemist = mol.intor("int2e")
h2_ao = np.einsum('ijkl->iklj', h2_chemist)

# Find transform from atomic orbitals (AO) to orthonormal orbitals (OO)
overlap = mol.intor("int1e_ovlp")
eigval, eigvec = np.linalg.eigh(overlap)
ao2oo = eigvec/np.sqrt(eigval[None, :])
h1_oo = np.einsum('mn,mi,nj->ij', h1_ao, ao2oo, ao2oo)

# Find transform from orthonormal orbitals (OO) to moleculeecular orbital (MO) which diagonalize h1
eig_value_h1_oo, eig_vector_h1_oo = np.linalg.eigh(h1_oo)
order = np.argsort(eig_value_h1_oo)
oo2mo = eig_vector_h1_oo[:, order]
ao2mo = ao2oo @ oo2mo

coord = np.zeros((200,3))
coord[:,2] = np.linspace(-3,3,200)

gto_val = mol.eval_gto("GTOval", coord)
plt.plot(coord[:,2], gto_val@ao2mo)
plt.xlabel("$z$")
plt.ylabel("$\\phi(z)$")
plt.savefig("molecular_orbitals.pdf")

