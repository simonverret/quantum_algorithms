import numpy as np
from pauli_strings import PauliString, Operator

class JordanWigner:
    def __init__(self, n):
        self.cdag = []
        self.c = []
        self.n = n
        for i in range(n):
            x_str = 'I'*(n-i-1)+'X'+'Z'*i  # III..X..ZZ
            y_str = 'I'*(n-i-1)+'Y'+'Z'*i  # III..X..ZZ
            self.cdag.append(.5*PauliString.from_str(x_str)+.5j*PauliString.from_str(y_str))
            self.c.append(.5*PauliString.from_str(x_str)-.5j*PauliString.from_str(y_str))
    
    def hamiltonian_operator(self, h1, h2):
        coefs = []
        pstrs = []
        for i in range(self.n):
            for j in range(self.n):
                cdagc_ij = self.cdag[i]*self.c[j]
                coefs.extend(h1[i,j]*cdagc_ij.coefs)
                pstrs.extend(cdagc_ij.pstrs)
                for k in range(self.n):
                    for l in range(self.n):
                        cdagcdagcc_ijkl = self.cdag[i]*self.cdag[j]*self.c[k]*self.c[l]
                        coefs.extend(0.5*h2[i, j, k, l]*cdagcdagcc_ijkl.coefs)
                        pstrs.extend(cdagcdagcc_ijkl.pstrs)

        return Operator(coefs, pstrs).combine().apply_threshold().sort()


def run_tests():
    np.set_printoptions(linewidth=1000)

    mapping = JordanWigner(4)

    print(len(mapping.cdag), 'creation operators')
    for ap in mapping.cdag:
        print(ap)
    
    print(len(mapping.c), 'annihilation operators')
    for am in mapping.c:
        print(am)

    print(len(mapping.cdag), 'creation operators matrix')
    for ap in mapping.cdag:
        print(ap.to_matrix().real)
    
    print(len(mapping.c), 'annihilation operators matrix')
    for am in mapping.c:
        print(am.to_matrix().real)

    ## Testing if the hamiltonian function is working required the H2 integrals. An independant test would be nice! 

if __name__ == "__main__":
    run_tests()