from pyscf import gto
import numpy as np
import matplotlib.pyplot as plt

from jordan_wigner import JordanWigner

def get_hamiltonian_integrals(molecule):
    # one-electron hamiltonian
    kin_ao = molecule.intor("int1e_kin")
    nuc_ao = molecule.intor("int1e_nuc")
    h1_ao = nuc_ao + kin_ao

    # two-electron hamiltonian
    h2_chemist = molecule.intor("int2e")
    h2_ao = np.einsum('ijkl->iklj', h2_chemist)

    # Find transform from atomic orbitals (AO) to orthonormal orbitals (OO)
    overlap = molecule.intor("int1e_ovlp")
    eigval, eigvec = np.linalg.eigh(overlap)
    ao2oo = eigvec/np.sqrt(eigval[None, :])
    h1_oo = np.einsum('mn,mi,nj->ij', h1_ao, ao2oo, ao2oo)

    # Find transform from orthonormal orbitals (OO) to moleculeecular orbital (MO) which diagonalize h1
    eig_value_h1_oo, eig_vector_h1_oo = np.linalg.eigh(h1_oo)
    order = np.argsort(eig_value_h1_oo)
    oo2mo = eig_vector_h1_oo[:, order]
    ao2mo = ao2oo @ oo2mo

    # One-particle and two-particle hamiltonian in MO
    h1_mo = np.einsum('mn,mi,nj->ij', h1_ao, ao2mo, ao2mo)
    h2_mo = np.einsum('mnop,mi,nj,ok,pl->ijkl', h2_ao, ao2mo, ao2mo, ao2mo, ao2mo)
    
    # Add the spin dimensions
    h1_mo_with_spin = np.kron(np.eye(2), h1_mo)
    h2_spin_tensor = np.kron(np.eye(2)[:, None, None, :],np.eye(2)[None, :, :, None])
    h2_mo_with_spin = np.kron(h2_spin_tensor, h2_mo)
    
    return h1_mo_with_spin, h2_mo_with_spin


def dihydrogene_gs_energy(mapping, distance): #units in AA
    molecule = gto.M(atom = [['H', (0,0,-distance/2)],['H', (0,0,distance/2)]], basis = 'sto-3g')
    h1_mo_with_spin, h2_mo_with_spin = get_hamiltonian_integrals(molecule)
    h_lcps = mapping.hamiltonian_lcps(h1_mo_with_spin, h2_mo_with_spin)
    return np.linalg.eigvalsh(h_lcps.to_matrix()).min() + molecule.energy_nuc()


def run():
    np.set_printoptions(linewidth=1000)

    distance=0.735 #units in AA
    molecule = gto.M(atom = [['H', (0,0,-distance/2)],['H', (0,0,distance/2)]], basis = 'sto-3g')
    h1_mo_with_spin, h2_mo_with_spin = get_hamiltonian_integrals(molecule)

    ## This is more a test of Jordan Wigner
    mapping = JordanWigner(h1_mo_with_spin.shape[0])
    h_lcps = mapping.hamiltonian_lcps(h1_mo_with_spin, h2_mo_with_spin)

    print("LCPS")
    print(h_lcps)

    print("\nMatrix")
    print(np.round(h_lcps.to_matrix().real, 3))
    
    print("\nEnergy")
    dissociation_energy = np.linalg.eigvalsh(h_lcps.to_matrix()).min() + molecule.energy_nuc()
    print(dissociation_energy)
    
    print(f"\nplot\ndistance, energy")
    x = np.arange(0.3, 2.3, .05)
    y = []
    for distance in x:
        energy = dihydrogene_gs_energy(mapping, distance)
        y.append(energy)
        print(f"   {distance:.3f}, {energy:.3f}")
    plt.plot(x, y)
    plt.plot(0.735, dissociation_energy, 'o')
    plt.ylabel("energy")
    plt.xlabel("distance")
    plt.title("H2")
    plt.savefig("energy_vs_distance_H2.pdf")

if __name__ == "__main__":
    run()