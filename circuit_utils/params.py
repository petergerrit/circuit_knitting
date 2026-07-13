"""Parameters and circuit definitions for circuit knitting simulations."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_unitary

# Simulation parameters
Nqbits = 6
Ntsteps = 3
mid = Nqbits - 2
mass = 1.125
epsilon = 0.8


def superposition_state_prep(N):
    """Prepare superposition state for the lattice."""
    circuit = QuantumCircuit(2*N)
    for n in range(0, mid, 1):
        if n % 2 == 1:
            circuit.h(n)
        if n > 0 and (n-2) % 4 == 0:
            circuit.x(n)
    circuit.h(mid)
    circuit.x([mid+1, mid+2])
    circuit.cx(mid, mid+1)
    circuit.cx(mid+1, mid+2)
    circuit.h(mid+1)
    for n in range(mid+3, 2*N, 1):
        if n % 2 == 1:
            circuit.h(n)
        if (n-2) % 4 == 0:
            circuit.x(n)
    U_sp = circuit.to_gate()
    U_sp.name = "U$_{sp}$"
    return U_sp


def meson_operator():
    """Create meson operator circuit."""
    circuit = QuantumCircuit(3)
    circuit.x(0)
    circuit.z(1)
    circuit.x(2)
    U_meson = circuit.to_gate()
    U_meson.name = "U$_{meson}$"
    return U_meson


def trotter1(N, T, epsilon, mass):
    """Create Trotter step 1 circuit."""
    circuit = QuantumCircuit(2*N)
    for t in range(T):
        for n in range(0, 2*N, 2):
            circuit.append(fermion_mass(epsilon, mass, (-1)**(n/2+1)), [n])

        for l in range(1, 2*N, 2):
            circuit.append(gauge_kinetic(epsilon), [l])

        for n in range(0, 2*N, 4):
            circuit.append(fermion_hopping(epsilon, 1), [n, n+1, (n+2) % (2*N)])

        for n in range(2, 2*N, 4):
            circuit.append(fermion_hopping(epsilon, 1), [n, n+1, (n+2) % (2*N)])

    U_trotter = circuit.to_gate()
    U_trotter.name = "U$_{T="+str(T)+"}$"
    return U_trotter


def gauge_kinetic(epsilon):
    """Gauge kinetic term."""
    circuit = QuantumCircuit(1)
    circuit.rx(epsilon/2, 0)
    U_kg = circuit.to_gate()
    U_kg.name = "U$_{gk}$"
    return U_kg


def fermion_mass(epsilon, mass, eta):
    """Fermion mass term."""
    circuit = QuantumCircuit(1)
    circuit.rz(-epsilon*mass * eta/2, 0)
    U_m = circuit.to_gate()
    U_m.name = "U$_m$"
    return U_m


def fermion_hopping(epsilon, eta):
    """Fermion hopping term."""
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
    U_fh.name = "U$_{fh}$"
    return U_fh


def trotter_stepper(Ntsteps, Nqbits, epsilon, mass, mid):
    """Create a Trotter stepper circuit."""
    for i in range(Ntsteps+1):
        qc = QuantumCircuit(2*Nqbits, 0)
        qc.append(superposition_state_prep(Nqbits), range(2*Nqbits))
        qc.append(trotter1(Nqbits, i, epsilon, mass), range(2*Nqbits))
        qc.append(meson_operator(), [mid, mid+1, mid+2])
    return qc


# Pre-defined circuits for epsilon=0.8
# These are the circuits used in the simulations
trot_step_1 = trotter_stepper(1, Nqbits, epsilon, mass, mid).decompose().decompose()
trot_step_2 = trotter_stepper(2, Nqbits, epsilon, mass, mid).decompose().decompose()
trot_step_1.measure_all()
trot_step_2.measure_all()
