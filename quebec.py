from qiskit_ibm_runtime.fake_provider import FakeQuebec
backend = FakeQuebec()
target = backend.target
cm = target.build_coupling_map()
cm.draw()
