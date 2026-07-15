"""
Trotter stepper implementation for quantum circuit evolution.
Extracted from functions.ipynb in the parent circuit_knitting directory.
"""
from qiskit import QuantumCircuit


def gauge_kinetic(epsilon):
    """Gauge kinetic term: exp[-i * epsilon/2 * sigma_x]"""
    circuit = QuantumCircuit(1)
    circuit.rx(epsilon/2, 0)
    U_kg = circuit.to_gate()
    U_kg.name = "U_gk"
    return U_kg


def fermion_mass(epsilon, mass, eta):
    """Fermion mass term: exp[-i * epsilon * mass * eta/2 * sigma_z]"""
    circuit = QuantumCircuit(1)
    circuit.rz(-epsilon*mass * eta/2, 0)
    U_m = circuit.to_gate()
    U_m.name = "U_m"
    return U_m


def fermion_hopping(epsilon, eta):
    """Fermion hopping term for 3 qubits."""
    circuit = QuantumCircuit(3)
    circuit.sxdg(1)
    circuit.s(1)
    circuit.cx(0, 1)
    circuit.sx(0)
    circuit.cx(0, 2)
    circuit.rx(-epsilon/4 * eta, 0)
    circuit.ry(-epsilon/4 * eta, 2)
    circuit.cx(0, 2)
    circuit.sxdg(0)
    circuit.cx(0, 1)
    circuit.sdg(1)
    circuit.sx(1)
    U_fh = circuit.to_gate()
    U_fh.name = "U_fh"
    return U_fh


def superposition_state_prep(N):
    """Prepare initial superposition state.
    
    Parameters:
        N: Number of qubits
    """
    circuit = QuantumCircuit(N)
    insertion_point = 4
    for n in range(0, insertion_point, 1):
        if n % 2 == 1:
            circuit.h(n)
        if n > 0 and (n-2) % 4 == 0:
            circuit.x(n)
    circuit.h(insertion_point)
    circuit.x([insertion_point+1, insertion_point+2])
    circuit.cx(insertion_point, insertion_point+1)
    circuit.cx(insertion_point+1, insertion_point+2)
    circuit.h(insertion_point+1)
    for n in range(insertion_point+3, N, 1):
        if n % 2 == 1:
            circuit.h(n)
        if (n-2) % 4 == 0:
            circuit.x(n)
    U_sp = circuit.to_gate()
    U_sp.name = "U_sp"
    return U_sp


def meson_operator():
    """Meson operator: X-Z-X on 3 qubits."""
    circuit = QuantumCircuit(3)
    circuit.x(0)
    circuit.z(1)
    circuit.x(2)
    U_meson = circuit.to_gate()
    U_meson.name = "U_meson"
    return U_meson


def trotter1(N, T, epsilon, mass):
    """
    Build a single Trotter step circuit.
    
    Parameters:
        N: Number of qubits
        T: Time step index
        epsilon: Trotter step size
        mass: Fermion mass parameter
    
    Returns:
        Quantum circuit as a gate
    """
    circuit = QuantumCircuit(N)
    for t in range(T):
        for n in range(0, N, 2):
            circuit.append(fermion_mass(epsilon, mass, (-1)**(n/2+1)), [n])
        
        for l in range(1, N, 2):
            circuit.append(gauge_kinetic(epsilon), [l])
    
        for n in range(0, N, 4):
            circuit.append(fermion_hopping(epsilon, 1), [n, n+1, (n+2) % N])
    
        for n in range(2, N, 4):
            circuit.append(fermion_hopping(epsilon, 1), [n, n+1, (n+2) % N])
    
    U_trotter = circuit.to_gate()
    U_trotter.name = f"U_T={T}"
    return U_trotter
