import numpy as np
import scipy.sparse as sparse
import random
from numpy import pi
# qiskit dependencies only used to compare results
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi



'''
The following variables contain the matrices that represent the buidling
blocks of quantum gates that my simulation can handle.

                --Single Qubit (Atomic) Gates--

'H' -> Hadamard Gate

'X' -> Pauli X Matrix (NOT Matrix) rotates state vector by pi about x-axis.

'Y' -> Pauli Y Matrix rotates state vector by pi about y-axis.

'Z' -> Pauli Z Matrix rotates state vector by pi about z-axis.

'I' -> Identity Matrix (Pauli 0 Matrix) leaves an arbitrary qubit unchanged.
        Primarily useful for constructing unitary matrix representations
        of an operation on a qubit that leaves other qubits unchanged.
        This can be realized by taking the tensor (Kronecker) product of an 
        arbitrary gate with identity matrices in the position of the 
        'unchanged' qubits.

                        --2 Qubit Gates--
'S' -> Swaps the state vector by flipping the position of any two qubits.
    i.e. S(1, 2) |01010> = |00110>

'CNOT' -> Controlled NOT gate (controlled X gate) maps the state in the second
          position to the opposite value if the first state is '1'.
          CNOT |00> = |00>
          CNOT |01> = |01>
          CNOT |10> = |11>
          CNOT |11> = |10>
'''
H = (1/np.sqrt(2)) * sparse.lil_array([[1, 1], [1, -1]],dtype=complex)
X = sparse.lil_array([[0,1], [1,0]],dtype=complex)
Y = sparse.lil_array([[0,-1j], [1j, 0]])
Z = sparse.lil_array([[1, 0], [0, -1]],dtype=complex)
S = sparse.lil_array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                      [0, 0, 0, 1]],dtype=complex)
I = sparse.eye(2, dtype=complex)
CNOT = sparse.lil_array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                         [0, 0, 1, 0]],dtype=complex)


'''
Leaves |0> unchanged, but adds a phase to |1> by mapping |1> to 
 exp(1j*theta)|1>.
'''
def Phase(theta):
    return sparse.lil_array([[1, 0], [0, np.exp(1j * theta)]])


'''
'CPHASE' is the controlled version of the phase gate. This gate and the 'phase'
gate are related to each other in an analogous way to 'CNOT' and 'X'. 
It takes in an angle theta as its only parameter.
    CPHASE(t) |00> = |00>
    CPHASE(t) |01> = |01>
    CPHASE(t) |10> = |10>
    CPHASE(t) |11> = exp(1j * t) * |00>
'''
def CPHASE(theta):
    return sparse.lil_array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                         [0, 0, 0, np.exp(1j * theta)]])


'''
Returns the unitary matrix that swaps qubits 's1' and 's2' in a circuit 
with a total of 'n_wires' qubits. As each basis state is mapped to one
unique basis element, this matrix consists of exactly 2**n_wires 1's.
'''
def swap(n_wires, s1, s2):
    if (s1 == s2): # Swapping a qubit with itself is trivial.
        return sparse.eye(2**n_wires,dtype=complex)
    else:
        s1, s2 = sorted((s1, s2))
        values = np.ones(2**n_wires)
        row = np.zeros(2**n_wires)
        col = np.zeros(2**n_wires)
        for i in range(2**n_wires): # swaps each state's binary digits
            binary = dec_to_binary(i, n_wires)
            new_binary = binary[0:s1] + binary[s2] + binary[s1+1:s2] +\
            binary[s1] + binary[s2 + 1::]
            j = int(new_binary, 2)
            row[i] = i # places element in location to map to transformed element
            col[i] = j
        return sparse.csr_matrix((values, (row, col)),
               shape=(2**n_wires, 2**n_wires))
    

'''
The function 'tensor_construct' creates and returns a sparse matrix 
representation of a particular gate.

                    ---Controlled Gates---
For controlled gates, the function starts by producing the unitary matrix 
corresponding to the controlled gate with 0 as the control qubit and 1
as the target qubit. This is achieved by taking the Kronecker product of the
2-qubit controlled gate with the identity matrix of the subpsace spanned by
qubits not being acted on (of dimension 2**(n_wires - 2) x 2**(n_wires - 2)).
Then two (or one) different matrices are used to swap the control with qubit 0
and the target with qubit 1.

The unitary matrix representing the desired gate, U can be obtained by
performing a change of basis (where swaps S1 S2 have occured) on a linear map 
(U': the representation of the controlled gate with a control of 0 and target
of 1).

U = (S1 @ S2) @ U' @ (S2 @ S1)

                    ---Single Qubit Gates---
The algorithm used to produce the unitary matrix representing a gate acting
on a single qubit is as follows. First, create the identity matrix
for all the states with an index less than the target qubit (I1) and then 
create an identity matrix for all the states with an index greater 
than the target (I2). If the single qubit gate can be represented by the
unitary matrix Us, the desired representation is the Kronecker product
'sandwich': U = I1 ⊗ Us ⊗ I2.



'''
def tensor_construct(gate, target, n_wires, **kwargs):
    single_qubit_dict = {"H":H, "X":X, "Y":Y, "Z":Z, "P":Phase}
    controlled_dict = {"X":CNOT, "P":CPHASE}
    if (gate == "S"):
        g = swap(n_wires, target, kwargs["Parameter"])
    elif "Control" in kwargs.keys():
        control = kwargs["Control"]
        if ("Parameter" in kwargs.keys()):
            g = controlled_dict[gate](kwargs["Parameter"])
        else:
            g = controlled_dict[gate]
        if (target != 0): # cases when control and 0 (c <-> 0) swapped first
            g = swap(n_wires, control, 0) @ swap(n_wires, target, 1) @\
            sparse.kron(g, sparse.eye(2**(n_wires - 2))) @\
            swap(n_wires, target, 1) @ swap(n_wires, control, 0)
        elif (control == 1): # only one swap matrix (1 <-> 0) is needed when c = 1 and t= 0
            g = swap(n_wires, 0, 1) @ sparse.kron(g, 
                sparse.eye(2**(n_wires - 2))) @ swap(n_wires, 0, 1)
        else: # if t=0, then (t <-> 1) must precede (c <-> 0) 
            g = swap(n_wires, target, 1) @ swap(n_wires, control, 0) @\
            sparse.kron(g, sparse.eye(2**(n_wires - 2))) @\
            swap(n_wires, control, 0) @ swap(n_wires, target, 1)
    else:
        if ("Parameter" in kwargs.keys()):
            g = single_qubit_dict[gate](kwargs["Parameter"])
        else:
            g = single_qubit_dict[gate]
        for i in range(0, target):
            g = sparse.kron(I, g) # kronecker products accounting for qubits less than target
        for i in range(target, n_wires - 1):
            g = sparse.kron(g, I) # kronecker products accounting for qubits greater than target
    return g
 

def dec_to_binary(dec, n_wires):
    binary = "{0:b}".format(int(dec))
    dl = n_wires - len(binary)
    binary = "0"*dl + binary # prepends 0's so string length matches number of qubits
    return binary


def dec_to_state(dec, n_wires):
    return sparse.csc_array((np.array([1]), ([dec], [0])), shape=(2**n_wires,1))


def basis_state_to_dec(state):
    return sparse.find(state)[0][0]


def get_gate(line, n_wires):
    kwargs = {}
    #print(line)
    if "@" in line: # finds parameter (if existing)
        gate = line[0:line.index("@")]
        parameter = line[line.index("@")+1:line.index(",")]
        kwargs.update({"Parameter":eval(parameter)})
    else:
        gate = line[0:line.index(",")]
    wires = line[line.index(",")+1::]
    if ("->" in line): # finds control and target (if gate is controlled)
        control = int(wires[:wires.index("->")].strip())
        target = int(wires[wires.index("->")+2:].strip())
        kwargs.update({"Control":control})
    else: # target when there is no control (a single gate)
        target = int(wires.strip())
    return tensor_construct(gate, target, n_wires, **kwargs)


def circuit_to_unitary(file_path):
    with open(file_path, 'r') as f:
        lines = []
        for line in f:
            if "N=" in line: # initializes unitary matrix
                get_numb = lambda x: x[x.index("=")+1:]
                n_wires = int(get_numb(line))
                matrix = sparse.eye(int(2**n_wires),dtype=complex,
                                    format="csr")
            else: # stores each gate in text format
                if "\n" in line:
                    line = line[:line.index("\n")].strip()
                lines.append(line)
    lines = lines[::-1] # order of matrix multiplication is reversed from how circuit is read
    for line in lines:
        matrix = matrix @ get_gate(line, n_wires)
    return matrix


'''
The function 'rand_circ' writes a random circuit to a file of name 'f_name'.
The following optional keyword arguments are supported:
    'n=': int, number of qubits
    'n_gates=': int, number of gates
    'control=True': bool, whether to include controlled gates

Here is an example of the plain text format I use to write quantum circuits:
    
    N=3
    H, 0
    Z, 2
    P@4.000969269573869, 0->2
    H, 1
    Z, 1
    P@4.781655085121643, 1
    S@1, 2

Each command is separated into a newline. Each circuit starts with the number
of qubits at the top as 'N='. A gate is represented by a string followed by a comma.
If the gate takes a parameter (like a phase), the gate name is followed by "@"
and then the value of the parameter. After the comma is a space and details 
about the qubit(s). If the gate acts on a single qubit, the index of the qubit
is stored. Otherwise, an arrow to the right of control index 
(and to the left of the target index) is stored.
'''


def bit_reverse_lines(start, end):
    diff = 1 + end - start
    lines = []
    for i in range(diff//2):
        lines.append("S@"+str(start + i)+", "+str(end - i))
    return lines


def rand_circ(f_name, **kwargs):
    if "n" in kwargs:
        n = kwargs["n"]
    else:
        n = random.randint(1, 10)
    if "n_gates" in kwargs:
        n_gates = kwargs["n_gates"]
    else:
        n_gates = random.randint(1,18)
    if "control" in kwargs or n == 1: # can't fit a control on 1 qubit circuit
        gate_set = ["H", "X", "Y", "Z", "P"]
    else:
        gate_set = ["H", "X", "Y", "Z", "P", "CX", "CP"]
    with open(f_name, 'w') as f:
        f.write("N="+str(n) + "\n")
        for i in range(n_gates):
            gate = random.choice(gate_set)
            if (gate[0] == "C"):
                control, target = random.sample(list(range(n)),2)
                wire = str(control) + "->" + str(target)
                gate = gate.replace("C", "")
            else:
                wire = str(random.choice(list(range(n))))
            if gate == "P":
                line = "P" + "@" + str(2 * np.pi * random.random())
            else:
                line = gate
            line += ", " + wire
            f.write(line+"\n")
    print(f"Task Completed: {f_name}")


'''
The following function translates my plain text circuits into a qiskit
circuit unitary matrix.
'''
def circuit_to_qiskit_unitary(file_path): 
    translation = {
        "H": "h",
        "X": "x",
        "Y": "y",
        "Z": "z",
        "P": "p"
    } # generates a string to be evaluated based on the gate and parameters
    with open(file_path, 'r') as f:
        lines = []
        for line in f:
            if "N=" in line:
                get_numb = lambda x: x[x.index("=")+1:]
                n_wires = int(get_numb(line))
            else:
                if "\n" in line:
                    line = line[:line.index("\n")].strip()
                lines.append(line)
    temp_circ = QuantumCircuit(n_wires)
    for line in lines:
        if "@" in line:
            gate = line[0:line.index("@")]
            parameter = line[line.index("@")+1:line.index(",")]
        else:
            gate = line[0:line.index(",")]
            parameter = None
        wires = line[line.index(",")+1::]
        if ("->" in line):
            c = int(wires[:wires.index("->")].strip())
            t = int(wires[wires.index("->")+2:].strip())
            if parameter == None:
                getattr(temp_circ, "c"+translation[gate])(c, t)
            else:
                getattr(temp_circ, "c"+translation[gate])(float(parameter),c, t)
        else:
            q = int(wires.strip())
            if parameter == None:
                getattr(temp_circ, translation[gate])(q)
            else:
                getattr(temp_circ, translation[gate])(float(parameter), q)
    return qi.Operator(temp_circ.reverse_bits()).data 
# Qiskit ordering is reversed for most-significant bit and least-significant bit


def random_tests(n_tests): # compares my matrices to those from Qiskit
    differences = []
    for i in range(n_tests):
        rand_circ("temp1_circ.txt")
        differences.append(np.abs(circuit_to_unitary("temp1_circ.txt").\
                toarray()-circuit_to_qiskit_unitary("temp1_circ.txt")).sum())
        if not np.allclose(circuit_to_unitary("temp1_circ.txt").toarray(),\
                           circuit_to_qiskit_unitary("temp1_circ.txt")):
            print("Failed")
            with open("temp1_circ.txt", "r") as f:
                for line in f:
                    print(line)
    print(max(differences)) # print largest absolute difference between
                            # corresponding matrix elements


# create unitary matrix representation for a circuit
def unitary_from_lines(n_qubs, lines):
    matrix = sparse.eye(int(2**n_qubs),dtype=complex,
                                    format="csr")
    for line in lines[::-1]: # matrix multiplication order needs to be reversed
        matrix = matrix @ get_gate(line, n_qubs)
    return matrix


# produces gates for inverse quantum Fourier transform
def iqft(n_qubits):
    n_qubits -= 1
    gates = []
    for i in range(n_qubits):
        gates.append("H, " + str(i))
        for j in range(0, n_qubits - i):
            phase = str(-np.pi / (2**(j+1)))
            gates.append("P@" + str(phase) + ", " + str(i) +\
                         "->" + str(1 + j + i))
    gates.append("H, " + str(n_qubits))
    gates += bit_reverse_lines(0, n_qubits)
    return gates


# produces gates for quantum Fourier transform
def qft(n_qubits):
    n_qubits -= 1
    gates = []
    gates += bit_reverse_lines(0, n_qubits)
    for i in range(n_qubits):
        gates.append("H, " + str(n_qubits - i))
        for j in range(n_qubits - i):
            phase = str(np.pi / (2**(j+1)))
            gates.append("P@" + str(phase) + ", " +\
            str(n_qubits - i) + "->" + str(n_qubits -  j - i - 1))
    gates.append("H, 0")
    return gates
