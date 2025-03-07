import numpy as np
from qiskit.circuit import QuantumCircuit

PAULI_LABELS = np.array(['I','Z','X','Y'])

PAULI_I = np.array([
        [  1,  0],
        [  0,  1]
    ], dtype=np.complex128)

PAULI_X = np.array([
        [  0,  1],
        [  1,  0]
    ], dtype=np.complex128)

PAULI_Y = np.array([
        [  0,-1j],
        [ 1j,  0]
    ], dtype=np.complex128)

PAULI_Z = np.array([
        [  1,  0],
        [  0, -1]
    ], dtype=np.complex128)


class PauliString:
    def __init__(self, z_bits, x_bits):
        self.z_bits = z_bits
        self.x_bits = x_bits

    def __str__(self):
        pauli_choices = np.flip(self.z_bits + 2*self.x_bits)
        return ''.join(PAULI_LABELS[pauli_choices])
    
    def __len__(self):
        return len(self.z_bits)

    def __mul__(self, other):
        if isinstance(other, PauliString):
            return self.mul_pauli_string(other)
        else:
            return self.mul_coef(other)

    def __rmul__(self, other):
        return self.__mul__(other)
    
    @classmethod
    def from_str(cls, pauli_str):
        char_array = np.array(list(reversed(pauli_str)))
        z_bits = np.logical_or(char_array=='Z', char_array=='Y')
        x_bits = np.logical_or(char_array=='X', char_array=='Y')
        return cls(z_bits, x_bits)

    @classmethod
    def from_zx_bits(cls, zx_bits):
        return cls(*np.split(zx_bits, 2))

    def to_zx_bits(self):
        return np.concat([self.z_bits, self.x_bits])

    def to_xz_bits(self):
        return np.concat([self.x_bits, self.z_bits])

    def ids(self):
        return np.logical_not(np.logical_or(self.x_bits, self.z_bits))
    
    def mul_pauli_string(self, other):
        new_z_bits = np.logical_xor(self.z_bits, other.z_bits)
        new_x_bits = np.logical_xor(self.x_bits, other.x_bits)
        w = 2*np.sum(self.x_bits*other.z_bits) + np.sum(self.z_bits*self.x_bits) + np.sum(other.x_bits*other.z_bits) - np.sum(new_z_bits*new_x_bits)
        phase = (-1j)**w

        return PauliString(new_z_bits, new_x_bits), phase

    def to_matrix(self):
        pauli_string_labels_list = np.array(list(str(self)))
        out = 1
        for pmat in pauli_string_labels_list:
            match str(pmat):
                case 'I': out = np.kron(out, PAULI_I)
                case 'X': out = np.kron(out, PAULI_X)
                case 'Y': out = np.kron(out, PAULI_Y)
                case 'Z': out = np.kron(out, PAULI_Z)
        return out
    
    def mul_coef(self, other):
        return LinearCombinaisonPauliString([other,], [self,])
    
    def diagonal(self):
        new_z_bits = np.logical_or(self.x_bits, self.z_bits)
        return PauliString(new_z_bits, np.zeros_like(new_z_bits))
    
    def circuit(self):
        circuit = QuantumCircuit(len(self))    
        for i in range(len(self)):
            if self.x_bits[i]:
                if self.z_bits[i]:
                    circuit.s(i)
                circuit.h(i)
        return circuit

    def eigenvalue(self, bitstr):
        assert len(bitstr)==len(self)
        eigval = 0
        for i in range(len(bitstr)):
            if str(self)[i] in ['X','Y','Z'] and bitstr[i]=='1':
                eigval-=1
            else:
                eigval+=1
        return eigval
        
    


class LinearCombinaisonPauliString:
    def __init__(self, coefs, pstrs):
        self.coefs = np.array(coefs)
        self.pstrs = np.array(pstrs)
        assert len(self.coefs) == len(self.pstrs)
        
    def __str__(self):     
        if len(self.coefs) > 6:       
            return '\n'.join([f"({coef:.2f})*{pstr}" for coef, pstr in zip(self.coefs, self.pstrs)])
        else:
            return ' + '.join([f"({coef:.2f})*{pstr}" for coef, pstr in zip(self.coefs, self.pstrs)])
    
    def __add__(self, other):
        coefs = np.concat((self.coefs, other.coefs))
        pstrs = np.concat((self.pstrs, other.pstrs))
        return LinearCombinaisonPauliString(coefs, pstrs)
    
    def __sub__(self, other):
        coefs = np.concat((self.coefs, -other.coefs))
        pstrs = np.concat((self.pstrs, other.pstrs))
        return LinearCombinaisonPauliString(coefs, pstrs)

    def __mul__(self, other):
        if isinstance(other, LinearCombinaisonPauliString):
            return self.mul_lcps(other)
        else:
            return self.mul_coef(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def mul_lcps(self, other):
        pstrs = []
        coefs = []
        for coef1, pstr1 in zip(self.coefs, self.pstrs):
            for coef2, pstr2 in zip(other.coefs, other.pstrs):
                pstr, w = pstr1 * pstr2
                pstrs.append(pstr)
                coefs.append(coef1*coef2*w)
        return LinearCombinaisonPauliString(coefs, pstrs)
    
    def mul_coef(self, other):
        return LinearCombinaisonPauliString(other*self.coefs, self.pstrs)
        
    def __getitem__(self, key):
        coefs = [self.coefs[key]] if isinstance(key, int) else self.coefs[key]
        pstrs = [self.pstrs[key]] if isinstance(key, int) else self.pstrs[key]
        return LinearCombinaisonPauliString(coefs, pstrs)
    
    def to_zx_bits(self):
        return np.stack([pstr.to_zx_bits() for pstr in self.pstrs])

    def to_xz_bits(self):
        return np.stack([pstr.to_zx_bits() for pstr in self.pstrs])

    def ids(self):
        return np.stack([pstr.ids() for pstr in self.pstrs])
    
    def combine(self):
        coefs = []
        pstrs = []
        pstrs_strs = []
        self_pstrs_strs = np.array([str(pstr) for pstr in self.pstrs])
        for pstr in self.pstrs:
            if not str(pstr) in pstrs_strs:
                pstrs_strs.append(str(pstr))
                pstrs.append(pstr)
                coefs.append(self.coefs[np.argwhere(self_pstrs_strs == str(pstr))].sum())
        return LinearCombinaisonPauliString(coefs, pstrs)

    def apply_threshold(self, threshold=1e-6):
        coefs = []
        pstrs = []
        for coef, pstr in zip(self.coefs, self.pstrs):
            if np.abs(coef) > threshold:
                coefs.append(coef)
                pstrs.append(pstr)
        return LinearCombinaisonPauliString(coefs, pstrs)

    def sort(self):
        idx = np.argsort(self.coefs)
        return LinearCombinaisonPauliString(self.coefs[idx], self.pstrs[idx])
    
    def to_matrix(self):
        dim = 2**len(self.pstrs[0])
        out = np.zeros((dim,dim), dtype=np.complex128)
        for coef, pstr in zip(self.coefs, self.pstrs):
            out += coef*pstr.to_matrix()
        return out

    def expectation(self, counts, verbose=False):
        out = 0
        for count_dict, coef, pstr in zip(counts, self.coefs, self.pstrs):
            pavg = 0
            total = 0
            if verbose: print(pstr)
            for result, count in count_dict.items():
                eigval = pstr.eigenvalue(result)
                pavg += count*eigval
                total += count
                if verbose: print(' ', result, eigval, count)
            if verbose: print(" ", pavg/total)
            out += coef*pavg/total
        out /= len(self.pstrs)
        return out
        


def run_tests():
    np.set_printoptions(linewidth=1000)

    z_bits = np.array([0,1,0,1], dtype=bool)
    x_bits = np.array([0,0,1,1], dtype=bool)
    pauli_string = PauliString(z_bits,x_bits)
    print(pauli_string)

    z_bits = np.array([1,0,1,1], dtype=bool)
    x_bits = np.array([1,1,0,0], dtype=bool)
    print(PauliString(z_bits,x_bits))

    pauli_string = PauliString.from_str('YXZI')
    print(pauli_string)

    pauli_string = PauliString.from_str('YXZI')
    zx_bits = pauli_string.to_zx_bits()
    print(zx_bits)
    xz_bits = pauli_string.to_xz_bits()
    print(xz_bits)

    pauli_string = PauliString.from_str('YXZI')
    ids = pauli_string.ids()
    print(ids)

    z_bits = np.array([0,1,0,1],dtype = bool)
    x_bits = np.array([0,0,1,1],dtype = bool)
    zx_bits = np.concatenate((z_bits,x_bits))
    pauli_string = PauliString.from_zx_bits(zx_bits)
    print(pauli_string)

    bits_1 = np.array([0,1,0,1],dtype = bool)
    bits_2 = np.array([0,1,1,1],dtype = bool)
    print(bits_1 + bits_2)
    print(np.sum(bits_1))

    pauli_string_1 = PauliString.from_str('IYZZ')
    pauli_string_2 = PauliString.from_str('IIXZ')
    new_pauli_string, phase = pauli_string_1 * pauli_string_2
    print(new_pauli_string, phase)

    pauli_string_1 = PauliString.from_str('ZZZZ')
    pauli_string_2 = PauliString.from_str('XXXI')
    new_pauli_string, phase = pauli_string_1 * pauli_string_2
    print(new_pauli_string, phase)

    pauli_string = PauliString.from_str('ZX')
    print(pauli_string)
    matrix = pauli_string.to_matrix()
    print(matrix)

    coefs = np.array([0.5, 0.5])
    pauli_string_1 = PauliString.from_str('IIXZ')
    pauli_string_2 = PauliString.from_str('IYZZ')
    pauli_strings = np.array([pauli_string_1,pauli_string_2], dtype=PauliString)
    lcps = LinearCombinaisonPauliString(coefs, pauli_strings)
    print(lcps)

    lcps_single = 1*PauliString.from_str('IIXZ')
    print(lcps_single)

    lcps = 0.5*pauli_string_1 + 0.5*pauli_string_2
    print(lcps)

    lcps_1 = 1*PauliString.from_str('IIXZ')
    lcps_2 = 1*PauliString.from_str('IYZZ')
    new_lcps = lcps_1 * lcps_2
    print(new_lcps)

    lcps = 1*PauliString.from_str('IIIZ') + 1*PauliString.from_str('IIZI') + 1*PauliString.from_str('IZII') + 1*PauliString.from_str('ZIII')
    print(lcps[0])
    print(lcps[1:3])
    print(lcps[-1])

    print('zx_bits')
    zx_bits = lcps.to_zx_bits()
    print(zx_bits)

    print('xz_bits')
    xz_bits = lcps.to_xz_bits()
    print(xz_bits)

    print('ids')
    ids = lcps.ids()
    print(ids)

    lcps_1 = 1*PauliString.from_str('IIIZ') - 0.5*PauliString.from_str('IIZZ') 
    lcps_2 = 1*PauliString.from_str('ZZZI') + 0.5*PauliString.from_str('ZZII') 
    lcps_3 = lcps_1 * lcps_2
    print(lcps_3)

    lcps_combined = lcps_3.combine()
    print(lcps_combined)

    lcps = lcps_combined.apply_threshold()
    print(lcps)

    lcps = (lcps_1 + lcps_2).sort()
    print(lcps)

    small_lcps = 1*PauliString.from_str('ZZ') + 2*PauliString.from_str('XX')
    matrix = small_lcps.to_matrix()
    print(matrix)

    diagonal_observable = 2*PauliString.from_str('ZZZZ') + 1*PauliString.from_str('IIZZ')
    counts = [{'0110' : 50, '1001' : 50}, {'0110' : 50, '1001' : 50}]
    print(diagonal_observable.expectation(counts, verbose=True))
    
if __name__ == "__main__":
    run_tests()