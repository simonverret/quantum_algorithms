#%%
from pyscf import gto

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit import transpile

from jordan_wigner import JordanWigner
from energy_vs_distance_H2 import get_hamiltonian_integrals


#%% variable circuit
if __name__ == "__main__":
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
    pauli_string = PauliString.from_str('ZIXY')
    diagonalizing_circuit = pauli_string.circuit()
    diagonal_pauli_string = pauli_string.circuit()
    print(diagonal_pauli_string) #should be 'ZIZZ'
    diagonalizing_circuit.draw('mpl')


    # %%
    observable = 2*PauliString.from_str('ZXZX') + 1*PauliString.from_str('IIZZ')
    diagonal_observables = observable.diagonal() 
    diagonalizing_circuits = observable.circuits()

    for diagonal_observable in diagonal_observables:
        print(diagonal_observable)
        
    for circuit in diagonalizing_circuits:
        print(circuit.draw())


    # %%

    params = [0,]

    def get_expected_value_and_counts(
        observable, 
        state_circuit, 
        params = None,
        execute_opts = {'shots' : 1024, 'seed_simulator' : 1}
    ):
        diagonalizing_circuits = observable.circuits()
        if params is not None:
            state_circuit = state_circuit.assign_parameters(params)
            print(state_circuit.draw())

        circuits = []
        for diagonalizing_circuit in diagonalizing_circuits:
            circuit = QuantumCircuit(4)
            circuit.compose(state_circuit, inplace=True)
            circuit.barrier()
            circuit.compose(diagonalizing_circuit, inplace=True)
            circuit.measure_all()
            circuits.append(circuit)
        circuits[0].draw('mpl')

        compiled_circuits = transpile(circuits, qasm_simulator)
        sim_result = qasm_simulator.run(compiled_circuits, **execute_opts).result()
        counts = sim_result.get_counts()
        expectation = observable.expectation(counts, verbose=True)

        return expectation, counts

    observable = 2*PauliString.from_str('ZXZX') + 1*PauliString.from_str('IIZZ')
    varform = varform_4qubits_1param
    backend = qasm_simulator
    estimator = Estimator(varform, backend)
    state_circuit = estimator.prepare_state_circuit(params)

    get_expected_value_and_counts(observable, state_circuit)

    #%% H2

    distance=0.735 #units in AA
    molecule = gto.M(atom = [['H', (0,0,-distance/2)],['H', (0,0,distance/2)]], basis = 'sto-3g')
    h1_mo_with_spin, h2_mo_with_spin = get_hamiltonian_integrals(molecule)

    mapping = JordanWigner(h1_mo_with_spin.shape[0])
    h_lcps = mapping.hamiltonian_lcps(h1_mo_with_spin, h2_mo_with_spin)

    get_expected_value_and_counts(h_lcps, varform_4qubits_1param, params={'a':0})

    #%%
    get_expected_value_and_counts(h_lcps, varform_4qubits_3params, params={'a':0, 'b':0, 'c':0})
