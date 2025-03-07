#%%
from qiskit.circuit import QuantumCircuit

def example_2qbits_2params_quantum_circuit(theta,phi):
    qc  = QuantumCircuit(2)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    qc.cx(0,1)
    return qc

varform_qc = example_2qbits_2params_quantum_circuit
qc = varform_qc(1,2)
print(qc.draw())


# %% Qiskit parameters
from qiskit.circuit import Parameter

a = Parameter('a')
b = Parameter('b')
varform_qc = QuantumCircuit(2)
varform_qc.ry(a,0)
varform_qc.rz(b,0)
varform_qc.cx(0,1)
print(varform_qc.parameters)
print(varform_qc.draw())


#%% Various ways of setting params
param_dict = {"a": 1, "b": 2}
qc = varform_qc.assign_parameters(param_dict)
print(qc.draw())

param_dict = {"a": 1, "b": 2}
qc = varform_qc.assign_parameters(param_dict)
print(qc.draw())

param_values = [1,2]
param_dict = dict(zip(varform_qc.parameters, param_values))
print(param_dict)
qc = varform_qc.assign_parameters(param_dict)
print(qc.draw())


#%% Prepare state

varform_4qubits_1param = QuantumCircuit(4)
a = Parameter('a')
varform_4qubits_1param.x(0)
varform_4qubits_1param.ry(a, 1)
varform_4qubits_1param.cx(1,0)
varform_4qubits_1param.cx(0,2)
varform_4qubits_1param.cx(1,3)

varform_4qubits_1param.draw('mpl')


#%%
varform_4qubits_3params = QuantumCircuit(4)
a = Parameter('a')
b = Parameter('b')
c = Parameter('c')
varform_4qubits_3params.x(0)
varform_4qubits_3params.x(2)
varform_4qubits_3params.barrier()
varform_4qubits_3params.ry(a,1)
varform_4qubits_3params.cx(1,3)
varform_4qubits_3params.ry(b,1)
varform_4qubits_3params.ry(c,3)
varform_4qubits_3params.cx(1,0)
varform_4qubits_3params.cx(3,2)

varform_4qubits_3params.draw('mpl')


# %%
import numpy as np
from qiskit_aer import Aer
from pauli_strings import PauliString

qasm_simulator = Aer.get_backend('qasm_simulator')

class Estimator:
    def __init__(self, varform, backend):
        self.varform = varform
        self.backend = backend

    def prepare_state_circuit(self, params):
        param_dict = dict(zip(self.varform.parameters, param_values))
        return self.varform.assign_parameters(param_dict)



varform = varform_4qubits_1param
backend = qasm_simulator
estimator = Estimator(varform,backend)
params = np.random.random(1)
state_circuit = estimator.prepare_state_circuit(params)
state_circuit.draw('mpl')


#%%
def diagonalizing_pauli_string_circuit(pauli_string):
    diagonalizing_circuit = QuantumCircuit(len(pauli_string))
    
    new_z_bits = np.logical_or(pauli_string.x_bits, pauli_string.z_bits)
    diagonal_pauli_string = PauliString(new_z_bits, np.zeros_like(new_z_bits))
    
    for i in range(len(pauli_string)):
        if pauli_string.x_bits[i]:
            if pauli_string.z_bits[i]:
                diagonalizing_circuit.s(i)
            diagonalizing_circuit.h(i)

    return diagonalizing_circuit, diagonal_pauli_string


pauli_string = PauliString.from_str('ZIXY')
diagonalizing_circuit = pauli_string.circuit()
diagonal_pauli_string = pauli_string.circuit()
print(diagonal_pauli_string) #should be 'ZIZZ'
diagonalizing_circuit.draw('mpl')


# %%
def diagonal_observables_and_circuits(observable):
    observables = []
    circuits = []
    for coefficient, pauli_string in zip(observable.coefs, observable.pstrs):
        diagonalizing_circuit = pauli_string.circuit()
        diagonal_pauli_string = pauli_string.diagonal()
        observables.append(coefficient*diagonal_pauli_string)
        circuits.append(diagonalizing_circuit)
    return observables, circuits


observable = 2*PauliString.from_str('ZXZX') + 1*PauliString.from_str('IIZZ')
diagonal_observables, diagonalizing_circuits = diagonal_observables_and_circuits(observable)

for diagonal_observable in diagonal_observables:
    print(diagonal_observable)
    
for circuit in diagonalizing_circuits:
    print(circuit.draw())


# %%
varform = varform_4qubits_1param
backend = qasm_simulator
estimator = Estimator(varform, backend)

observable = 2*PauliString.from_str('ZXZX') + 1*PauliString.from_str('IIZZ')

# estimator.set_observable(observable)
diagonal_observables, diagonalizing_circuits = diagonal_observables_and_circuits(observable)

params = [0,]
state_circuit = estimator.prepare_state_circuit(params)

# circuits = estimator.assemble_circuits(state_circuit)\
circuits = []
for diagonalizing_circuit in diagonalizing_circuits:
    circuit = QuantumCircuit(4)
    circuit.compose(state_circuit, inplace=True)
    circuit.barrier()
    circuit.compose(diagonalizing_circuit, inplace=True)
    circuit.measure_all()
    circuits.append(circuit)
circuits[0].draw('mpl')

#%%
from qiskit import transpile

compiled_circuit = transpile(circuits, qasm_simulator)
execute_opts = {'shots' : 1024, 'seed_simulator' : 1}
sim_result = qasm_simulator.run(compiled_circuit, **execute_opts).result()
counts = sim_result.get_counts()

observable.expectation(counts, verbose=True)



#%% idea
# pstr.diagonal()
# pstr.circuit()
# pstr.eigenvalue(result)

