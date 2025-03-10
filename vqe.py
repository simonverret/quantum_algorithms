#%%
import numpy as np
from pyscf import gto
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit import transpile
from qiskit_aer import Aer

from pauli_strings import PauliString
from jordan_wigner import JordanWigner
from energy_vs_distance_H2 import get_hamiltonian_integrals, dihydrogene_gs_energy

QASM = Aer.get_backend('qasm_simulator')
QASM_OPTIONS = {'shots' : 1024, 'seed_simulator' : 1}
SLSQP_OPTIONS =  {'maxiter' : 100,'eps' : 1e-1, 'ftol' : 1e-4, 'disp' : True, 'iprint' : 1}

A = Parameter('a')
B = Parameter('b')
C = Parameter('c')

STATE_PREP_4_1 = QuantumCircuit(4)
STATE_PREP_4_1.x(0)
STATE_PREP_4_1.ry(A, 1)
STATE_PREP_4_1.cx(1,0)
STATE_PREP_4_1.cx(0,2)
STATE_PREP_4_1.cx(1,3)

STATE_PREP_4_3 = QuantumCircuit(4)
STATE_PREP_4_3.x(0)
STATE_PREP_4_3.x(2)
STATE_PREP_4_3.barrier()
STATE_PREP_4_3.ry(A,1)
STATE_PREP_4_3.cx(1,3)
STATE_PREP_4_3.ry(B,1)
STATE_PREP_4_3.ry(C,3)
STATE_PREP_4_3.cx(1,0)
STATE_PREP_4_3.cx(3,2)


def expected_value(observable, state_circuit, params=[]):    
    state_circuit = state_circuit.assign_parameters(params)

    circuits = []
    for obs_circuit in observable.circuits():
        circuit = QuantumCircuit(4)
        circuit.compose(state_circuit, inplace=True)
        # circuit.barrier()
        circuit.compose(obs_circuit, inplace=True)
        circuit.measure_all()
        circuits.append(circuit)

    compiled_circuits = transpile(circuits, QASM)
    sim_result = QASM.run(compiled_circuits, **QASM_OPTIONS).result()
    counts = sim_result.get_counts()
    expectation = observable.expectation(counts)
    return expectation.real


def vqe_energy(hamiltonian, state_circuit):
    def quantum_eigensolver(params):
        return expected_value(hamiltonian, state_circuit, params)
    
    minimization_result = minimize(
        quantum_eigensolver,
        np.zeros(len(state_circuit.parameters)),
        method = 'SLSQP',
        options = SLSQP_OPTIONS
    )
    return minimization_result.fun


def vqe_H2(mapping, distance, state_circuit):
    molecule = gto.M(atom = [['H', (0,0,-distance/2)],['H', (0,0,distance/2)]], basis='sto-3g')
    h1_mo_with_spin, h2_mo_with_spin = get_hamiltonian_integrals(molecule)
    h_lcps = mapping.hamiltonian_lcps(h1_mo_with_spin, h2_mo_with_spin)
    return vqe_energy(h_lcps, state_circuit) + molecule.energy_nuc()


if __name__ == "__main__":

    mapping = JordanWigner(4)
    x = np.arange(0.3, 2.3, .05)
    y1 = []
    y2 = []

    print(f"\nplot\ndistance, VQE energy, target")
    for distance in x:
        energy = vqe_H2(mapping, distance, STATE_PREP_4_3)
        true_energy = dihydrogene_gs_energy(mapping, distance)
        y1.append(energy)
        y2.append(true_energy)
        print(f"   {distance:.3f}, {energy:.3f}, {true_energy:.3f}")
    plt.plot(x, y1, label="VQE")
    plt.plot(x, y2, label="exact")
    plt.ylabel("energy")
    plt.xlabel("distance")
    plt.title("H2")
    plt.show()
    # plt.savefig("energy_vs_distance_H2.pdf")
