"""Batch processing utilities for heralded erasure decoder circuits.

This module provides optimized batch building of Clifford circuits from erasure syndromes.
"""

from stimdecoders.utils import codes, circuits, bitops
import stim
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class BatchCircuitBuilder:
    """Optimized batch builder for Clifford circuits from erasure syndromes.
    
    This class pre-computes invariant circuit components and uses templates to
    quickly build circuits for multiple syndromes. Supports both RSC and RepetitionCode.
    """
    
    def __init__(self, erasure_circuit: circuits.Circuit):
        """Initialize the batch builder.
        
        Args:
            erasure_circuit: The erasure or hybrid circuit (RSC or RepetitionCode)
        """
        self.code = erasure_circuit.code
        self.erasure_circuit = erasure_circuit
        self.noise_dict = erasure_circuit.noise_model.noise_dict
        
        # Determine code type
        self.is_rsc = isinstance(self.code, codes.RSC)
        self.is_repetition = isinstance(self.code, codes.RepetitionCode)
        
        if not (self.is_rsc or self.is_repetition):
            raise TypeError("BatchCircuitBuilder supports RSC and RepetitionCode only.")
        
        # RSC-specific initialization
        if self.is_rsc:
            self.meas_sets, self.meas_sets_norms = erasure_circuit.get_measurement_sets()
            self.erasure_bitmask = erasure_circuit.erasure_bitmask

        # Pre-compute circuit template components
        self._precompute_template()
        
    def _precompute_template(self):
        """Pre-compute the invariant parts of the circuit."""
        if self.is_rsc:
            self._precompute_template_rsc()
        elif self.is_repetition:
            self._precompute_template_repetition()
    
    def _precompute_template_rsc(self):
        """Pre-compute template for RSC codes."""
        # Store indices for each measurement set segment
        self.meas_indices = []
        curr_idx = 0
        for norm in self.meas_sets_norms:
            self.meas_indices.append((curr_idx, curr_idx + norm))
            curr_idx += norm
        
        # Pre-compute which stages are active (have erasure errors possible or hybrid noise)
        # For hybrid circuits, we need to process stages even if no erasures occur
        # because we still apply depolarizing noise to unerased qubits
        # We need to process if p > 0 (there's depolarizing noise to apply, even if all goes to erasure)
        has_depolarizing_noise = self.noise_dict.get('p', 0) > 0
        
        self.active_stages = {
            'sp': self.code.sp_support & self.erasure_bitmask > 0 and self.noise_dict.get('sp-e', 0) > 0,
            'hadamard1': self.code.hadamard_support & self.erasure_bitmask > 0 and self.noise_dict.get('sqg-e', 0) > 0,
            'cnots': [((self.code.cnot_bitmasks[i] >> self.code.eq_diff) & self.erasure_bitmask > 0 and
                      self.noise_dict.get('tqg-e', 0) > 0) or has_depolarizing_noise for i in range(4)],
            'hadamard2': self.code.hadamard_support & self.erasure_bitmask > 0 and self.noise_dict.get('sqg-e', 0) > 0,
            'ancilla_meas': self.code.ancilla_measure_support & self.erasure_bitmask > 0 and
                           self.noise_dict.get('meas-e', 0) > 0,
        }
        
        # Pre-compute CNOT support bitmasks (qubits involved in each CNOT round)
        # These are the original qubit IDs (not shifted by eq_diff)
        self.cnot_support_bitmasks = [
            self.code.cnot_bitmasks[i] >> self.code.eq_diff for i in range(4)
        ]
        
        # Pre-build the base circuit string parts (parts that don't depend on syndrome)
        self.base_reset = self.erasure_circuit.cached_strings["sp"]
        self.base_hadamard = self.erasure_circuit.cached_strings["h"]
        self.base_cnots = [self.erasure_circuit.cached_strings.get(f"cnot_{i}") if self.erasure_circuit.cached_strings.get(f"cnot_{i}", None) else f"CX {' '.join(f'{g[0]} {g[1]}' for g in gate_check)}"
                          for i, gate_check in enumerate(self.code.gates)]
        self.base_measurements = self.erasure_circuit.cached_strings["meas"]
        self.base_data_measurements = self.erasure_circuit.cached_strings["dmeas"]
        
        # Pre-cache Pauli error probabilities
        self.p_tqg_pauli = self.noise_dict.get('tqg', 0)
        p = self.noise_dict.get('p', 0)
        f = self.noise_dict.get('f', 0)
        self.p_unerased = p * (1 - f)
    
    def _precompute_template_repetition(self):
        """Pre-compute template for Repetition codes."""
        # Base circuit strings
        self.base_reset = self.erasure_circuit.cached_strings["sp"]
        self.base_cnots = [self.erasure_circuit.cached_strings[f"cnot_{i}"] for i in range(2)]
        self.base_measurements = self.erasure_circuit.cached_strings["meas"]
        self.base_data_measurements = self.erasure_circuit.cached_strings["dmeas"]
        
        # Pre-compute Pauli probabilities
        p = self.noise_dict.get('p', 0)
        f = self.noise_dict.get('f', 0)
        self.p_unerased = p * (1 - f)
        self.p_erased = 0.75
        
        # Number of qubits measured per round (for indexing into syndrome)
        self.qubits_per_round = self.code.num_qubits - 1

    def _extract_erasures_vectorized(self, syndromes: np.ndarray):
        """Extract erasure qubit indices from syndromes in a vectorized manner.
        
        Args:
            syndromes: Array of syndromes (num_syndromes, syndrome_length)
            
        Returns:
            List of patterns (format depends on code type)
        """
        if self.is_rsc:
            return self._extract_erasures_vectorized_rsc(syndromes)
        elif self.is_repetition:
            return self._extract_erasures_vectorized_repetition(syndromes)
    
    def _extract_erasures_vectorized_rsc(self, syndromes: np.ndarray) -> List[Dict[str, List[int]]]:
        """Extract erasure qubit indices from syndromes for RSC.
        
        For hybrid circuits, we need to track both erased and unerased qubits
        to apply appropriate depolarizing noise rates.
        """
        erasure_patterns = []
        
        for syndrome in syndromes:
            pattern = {}
            stage_idx = 0
            
            # State preparation erasures
            if self.active_stages['sp']:
                start, end = self.meas_indices[stage_idx]
                erased = [self.meas_sets[stage_idx][i] for i, m in enumerate(syndrome[start:end]) if m]
                pattern['sp'] = erased if erased else []
                stage_idx += 1
            else:
                pattern['sp'] = []
            
            # First Hadamard erasures
            if self.active_stages['hadamard1']:
                start, end = self.meas_indices[stage_idx]
                erased = [self.meas_sets[stage_idx][i] for i, m in enumerate(syndrome[start:end]) if m]
                pattern['hadamard1'] = erased if erased else []
                stage_idx += 1
            else:
                pattern['hadamard1'] = []
            
            # CNOT erasures for each check
            pattern['cnots_erased'] = []
            pattern['cnots_unerased'] = []
            for check_num in range(4):
                if self.active_stages['cnots'][check_num]:
                    start, end = self.meas_indices[stage_idx]
                    # Each CNOT gate has 2 syndrome bits (control, target)
                    erased = []
                    unerased = []
                    
                    syndrome_slice = syndrome[start:end]
                    for i, (control, target) in enumerate(self.code.gates[check_num]):
                        control_erased = syndrome_slice[2*i]
                        target_erased = syndrome_slice[2*i + 1]
                        
                        if control_erased:
                            erased.append(control)
                        else:
                            unerased.append(control)
                        
                        if target_erased:
                            erased.append(target)
                        else:
                            unerased.append(target)
                    
                    # Remove duplicates while preserving efficiency
                    pattern['cnots_erased'].append(list(dict.fromkeys(erased)))
                    pattern['cnots_unerased'].append(list(dict.fromkeys(unerased)))
                    stage_idx += 1
                else:
                    pattern['cnots_erased'].append([])
                    pattern['cnots_unerased'].append([])
            
            # Second Hadamard erasures
            if self.active_stages['hadamard2']:
                start, end = self.meas_indices[stage_idx]
                erased = [self.meas_sets[stage_idx][i] for i, m in enumerate(syndrome[start:end]) if m]
                pattern['hadamard2'] = erased if erased else []
                stage_idx += 1
            else:
                pattern['hadamard2'] = []
            
            # Ancilla measurement erasures
            if self.active_stages['ancilla_meas']:
                start, end = self.meas_indices[stage_idx]
                erased = [self.meas_sets[stage_idx][i] for i, m in enumerate(syndrome[start:end]) if m]
                pattern['ancilla_meas'] = erased if erased else []
                stage_idx += 1
            else:
                pattern['ancilla_meas'] = []
            
            erasure_patterns.append(pattern)
        
        return erasure_patterns
    
    def _extract_erasures_vectorized_repetition(self, syndromes: np.ndarray) -> List[Tuple[List[List[int]], List[List[int]]]]:
        """Extract erased and unerased qubit indices from syndromes for RepetitionCode."""
        patterns = []
        
        for syndrome in syndromes:
            erased = [[], []]
            unerased = [[], []]
            
            sample_index = 0
            for gate_index in range(2):
                # Process each qubit in this CNOT round
                for i, result in enumerate(syndrome[sample_index:sample_index + self.qubits_per_round]):
                    qubit = self.code.gates[gate_index][i // 2][i % 2]
                    if result:
                        erased[gate_index].append(qubit)
                    else:
                        unerased[gate_index].append(qubit)
                
                sample_index += self.qubits_per_round
            
            patterns.append((erased, unerased))
        
        return patterns
    
    def _build_circuit_from_pattern(self, pattern) -> circuits.Circuit:
        """Build a circuit from a pre-computed erasure pattern.
        
        Args:
            pattern: Pattern data (format depends on code type)
            
        Returns:
            The constructed Circuit object
        """
        if self.is_rsc:
            return self._build_circuit_from_pattern_rsc(pattern)
        elif self.is_repetition:
            return self._build_circuit_from_pattern_repetition(pattern)
    
    def _build_circuit_from_pattern_rsc(self, pattern: Dict[str, List[int]]) -> circuits.Circuit:
        """Build a circuit from a pre-computed erasure pattern for RSC."""
        circuit = circuits.Circuit(code=self.code)
        
        # Reset
        circuit._circ_str.append(self.base_reset)
        if self.erasure_circuit.cached_strings.get("sp_pauli"):
            circuit._circ_str.append(self.erasure_circuit.cached_strings["sp_pauli"])
        
        # State preparation errors
        if pattern['sp']:
            circuit.add_depolarize1(pattern['sp'], p=0.75)
        
        # First Hadamard
        circuit._circ_str.append(self.base_hadamard)
        if self.erasure_circuit.cached_strings.get("h_pauli"):
            circuit._circ_str.append(self.erasure_circuit.cached_strings["h_pauli"])
        if pattern['hadamard1']:
            circuit.add_depolarize1(pattern['hadamard1'], p=0.75)
        
        # CNOTs
        for check_num in range(4):
            circuit._circ_str.append(self.base_cnots[check_num])
            if self.erasure_circuit.cached_strings.get(f"cnot_{check_num}_pauli"):
                circuit._circ_str.append(self.erasure_circuit.cached_strings[f"cnot_{check_num}_pauli"])
            elif (self.erasure_circuit.pauli_bitmask & (self.cnot_support_bitmasks[check_num])) > 0 and self.erasure_circuit.noise_model.noise_dict.get("tqg", 0) > 0:
                circuit.add_depolarize1(bitops.mask_iter_indices(self.erasure_circuit.pauli_bitmask & (self.cnot_support_bitmasks[check_num])), p=0.75)
            # Apply appropriate depolarizing noise based on erasure status
            if self.code.distance == 3:
                print(pattern['cnots_unerased'][check_num])
            if pattern['cnots_unerased'][check_num] and self.p_unerased > 0:
                circuit.add_depolarize1(pattern['cnots_unerased'][check_num], p=self.p_unerased)
            if pattern['cnots_erased'][check_num]:
                circuit.add_depolarize1(pattern['cnots_erased'][check_num], p=0.75)
        
        # Second Hadamard
        circuit._circ_str.append(self.base_hadamard)
        if self.erasure_circuit.cached_strings.get("h_pauli"):
            circuit._circ_str.append(self.erasure_circuit.cached_strings["h_pauli"])
        if pattern['hadamard2']:
            circuit.add_depolarize1(pattern['hadamard2'], p=0.75)
        
        # Measurements
        circuit._circ_str.append(self.base_measurements)
        if self.erasure_circuit.cached_strings.get("meas_pauli"):
            circuit._circ_str.append(self.erasure_circuit.cached_strings["meas_pauli"])
        if pattern['ancilla_meas']:
            circuit.add_depolarize1(pattern['ancilla_meas'], p=0.75)
        
        # Append cached strings (detectors and observables)
        circuit._circ_str.append(self.erasure_circuit.detector_cache[0])
        circuit._circ_str.append(self.base_data_measurements)
        circuit.append_to_circ_str(self.erasure_circuit.detector_cache[1 : 1 + self.code.num_ancillas//2])
        circuit._circ_str.append(self.erasure_circuit.cached_strings["obs_0"])

        return circuit
    
    def _build_circuit_from_pattern_repetition(self, pattern: Tuple[List[List[int]], List[List[int]]]) -> circuits.Circuit:
        """Build a circuit from a pre-computed pattern for RepetitionCode."""
        erased, unerased = pattern
        
        circuit = circuits.Circuit(code=self.code)
        circuit.set_noise_model(self.erasure_circuit.noise_model)
        
        # State preparation
        circuit._circ_str.append(self.base_reset)
        
        # CNOT rounds
        for gate_index in range(2):
            circuit._circ_str.append(self.base_cnots[gate_index])
            
            # Add depolarizing errors for unerased qubits
            if self.p_unerased > 0 and unerased[gate_index]:
                circuit.add_depolarize1(unerased[gate_index], self.p_unerased)
            
            # Add depolarizing errors for erased qubits (effective Pauli error after erasure)
            if erased[gate_index]:
                circuit.add_depolarize1(erased[gate_index], self.p_erased)
        
        # Measurements and detectors
        circuit._circ_str.append(self.base_measurements)
        circuit._circ_str.append(self.erasure_circuit.detector_cache[0])
        circuit._circ_str.append(self.base_data_measurements)
        circuit.append_to_circ_str(self.erasure_circuit.detector_cache[1: 1 + self.code.num_ancillas])
        circuit._circ_str.append(self.erasure_circuit.cached_strings["obs_0"])

        return circuit
    
    def build_batch(self, syndromes: np.ndarray) -> List[stim.Circuit]:
        """Build Clifford circuits for a batch of syndromes.
        
        Args:
            syndromes: Array of syndromes (num_syndromes, syndrome_length)
            
        Returns:
            List of stim.Circuit objects, one for each syndrome
        """
        # Extract erasure patterns in a vectorized manner
        patterns = self._extract_erasures_vectorized(syndromes)
        
        # Build circuits from patterns
        circuits_list = []
        for pattern in patterns:
            circuit = self._build_circuit_from_pattern(pattern)
            circuits_list.append(circuit.to_stim_circuit())
        
        return circuits_list
    
    def build_batch_with_cache(self, syndromes: np.ndarray, 
                               circuit_cache: Dict[bytes, stim.Circuit] = None) -> Tuple[List[stim.Circuit], Dict[bytes, stim.Circuit]]:
        """Build Clifford circuits for a batch with caching.
        
        Args:
            syndromes: Array of syndromes (num_syndromes, syndrome_length)
            circuit_cache: Optional cache dictionary to use/update
            
        Returns:
            Tuple of (list of circuits, updated cache)
        """
        if circuit_cache is None:
            circuit_cache = {}
        
        circuits_list = []
        patterns = self._extract_erasures_vectorized(syndromes)
        
        for syndrome, pattern in zip(syndromes, patterns):
            syndrome_key = syndrome.tobytes()
            
            if syndrome_key in circuit_cache:
                circuits_list.append(circuit_cache[syndrome_key])
            else:
                circuit = self._build_circuit_from_pattern(pattern)
                stim_circuit = circuit.to_stim_circuit()
                circuit_cache[syndrome_key] = stim_circuit
                circuits_list.append(stim_circuit)
        
        return circuits_list, circuit_cache


def batch_build_clifford_circuits(erasure_circuit: circuits.Circuit,
                                  syndromes: np.ndarray,
                                  use_cache: bool = True,
                                  circuit_cache: Dict[bytes, stim.Circuit] = None) -> Tuple[List[stim.Circuit], Dict[bytes, stim.Circuit]]:
    
    """Build Clifford circuits for multiple syndromes in batch mode.
    
    This is the main entry point for batch circuit building. It creates a
    BatchCircuitBuilder and uses it to efficiently build circuits.
    
    Args:
        erasure_circuit: The erasure or hybrid circuit (RSC or RepetitionCode)
        syndromes: Array of syndromes (num_syndromes, syndrome_length)
        use_cache: Whether to use caching
        circuit_cache: Optional pre-existing cache
        
    Returns:
        Tuple of (list of circuits, cache dictionary)
    """
    builder = BatchCircuitBuilder(erasure_circuit)
    
    if use_cache:
        return builder.build_batch_with_cache(syndromes, circuit_cache)
    else:
        circuits_list = builder.build_batch(syndromes)
        return circuits_list, {}
