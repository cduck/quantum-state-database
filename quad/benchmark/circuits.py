from __future__ import annotations

import dataclasses

import numpy as np
import cirq
from cirqtools import qutrit





@dataclasses.dataclass
class RotationNoise(cirq.NoiseModel):
    '''A symmetric depolarizing noise model.

    Randomly adds error after each single-qubit (two-qubit) gate with
    probability `p1` (`p2`).  The errors have equal chance of being X, Y, or Z.
    '''
    p1: float
    p2: float
    prng: np.random.RandomState = dataclasses.field(
        default_factory=np.random.RandomState)

    def noisy_operation(self, op: cirq.Operation) -> cirq.OP_TREE:
        yield op
        if not isinstance(op.gate, cirq.MeasurementGate):
            p = (self.p1, self.p2)[len(op.qubits) > 1]
            if p == 0:
                return
            for q in op.qubits:
                sx, sy, sz = self.prng.normal(0, p/3, 3)
                yield cirq.X(q) ** sx
                yield cirq.Y(q) ** sy
                yield cirq.Z(q) ** sz


@dataclasses.dataclass
class DepolarizingChannelNoise(cirq.NoiseModel):
    '''A symmetric depolarizing noise model.

    Adds a depolarizing error channel after each single-qubit (two-qubit) gate
    with probability `p1` (`p2`).  The errors have equal chance of being X, Y,
    or Z.
    '''
    p1: float
    p2: float
    prng: np.random.RandomState = dataclasses.field(
        default_factory=np.random.RandomState)

    def noisy_operation(self, op: cirq.Operation) -> cirq.OP_TREE:
        yield op
        if not isinstance(op.gate, cirq.MeasurementGate):
            p = (self.p1, self.p2)[len(op.qubits) > 1]
            if p == 0:
                return
            g = cirq.depolarize(p)
            for q in op.qubits:
                yield g.on(q)


@dataclasses.dataclass
class RandomCircuitFactory:
    '''A factory for random quantum circuits with a layered structure of
    two-qubit gates.
    '''
    qubits: Iterable[cirq.GridQubit] = tuple(cirq.GridQubit.rect(3,3))
    depth: int = 4
    two_qubit_op_factory: Callable[[cirq.GridQubit, cirq.GridQubit,
                                    np.random.RandomState], cirq.OP_TREE
                                  ] = lambda a, b, _: cirq.CZ(a, b)
    pattern: Sequence[cirq.experiments.GridInteractionLayer
                     ] = cirq.experiments.GRID_ALIGNED_PATTERN
    single_qubit_gates: Sequence[cirq.Gate
        ] = (cirq.X**0.5, cirq.Y**0.5,
             cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5))
    add_final_single_qubit_layer: bool = True
    noise_model: cirq.NoiseModel = RotationNoise(0.1, 0.1, None)

    def make(self, circuit_seed: Any, noise_seed: Any=None) -> cirq.Circuit:
        c = (cirq.experiments
                .random_rotations_between_grid_interaction_layers_circuit(
            qubits=self.qubits,
            depth=self.depth,
            two_qubit_op_factory=self.two_qubit_op_factory,
            pattern=self.pattern,
            single_qubit_gates=self.single_qubit_gates,
            add_final_single_qubit_layer=self.add_final_single_qubit_layer,
            seed=circuit_seed,
        ))
        if noise_seed is not None:
            prng = np.random.RandomState(noise_seed)
            try:
                self.noise_model.prng = prng
                return c.with_noise(self.noise_model)
            finally:
                self.noise_model.prng = None
        return c
