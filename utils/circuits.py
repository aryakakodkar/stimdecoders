from typing import List, Literal, Tuple
import numpy as np
import stim
import time

from stimdecoders.utils import bitops, codes, noise

CORRELATED_CNOT_CACHE = {}

def pauli_erasure_cnot_optimized(gates: List[tuple], erasure_bitmask: int, p: float, eq_diff: int = 0):
    """Builds optimized string list for symmetric Pauli + Erasure errors on CNOTs.

    Args:
        gates: List of (control, target) qubit index tuples for CNOTs.
        erasure_bitmask: Python int bitmask indicating which qubits have erasure errors.
        p: Probability of error occurring on each CNOT.
        eq_diff: Difference between data and fictitious ancilla qubit indices.

    Returns:
        List of strings representing the error operations to be appended to a Circuit._circ_str.
    """
    err_string = []
    PAULI_STR = ('I', 'X', 'Y', 'Z', 'J')
    
    for gate in gates:
        g0, g1 = gate[0], gate[1]
        erasure0 = (erasure_bitmask >> g0) & 1
        erasure1 = (erasure_bitmask >> g1) & 1
        gate_properties = (erasure0, erasure1, p)

        if gate_properties not in CORRELATED_CNOT_CACHE:
            # Build cache for this configuration
            else_space = 1.0
            num_errors = (4 + erasure0) * (4 + erasure1) - 1
            err_probs = p / num_errors
            
            cache_entries = []
            
            for q1 in range(4 + erasure0):
                for q2 in range(4 + erasure1):
                    if (q1, q2) == (0, 0): 
                        continue
                    
                    cache_entries.append((
                        q1, q2,
                        erasure0 and q1 != 0,
                        erasure1 and q2 != 0,
                        err_probs / else_space
                    ))
                    else_space -= err_probs
            
            CORRELATED_CNOT_CACHE[gate_properties] = cache_entries
        
        # Build error strings using precomputed cache
        cache_entries = CORRELATED_CNOT_CACHE[gate_properties]
        
        for i in range(len(cache_entries)):
            q1, q2, e0, e1, err_prob = cache_entries[i]
            
            parts = ""
            
            # Error prefix
            if i == 0:
                parts += f"CORRELATED_ERROR({err_prob}) "
            else:
                parts += f"ELSE_CORRELATED_ERROR({err_prob}) "

            # Pauli errors (skip identity and erasure marker)
            if 0 < q1 < 4:
                parts += f"{PAULI_STR[q1]}{g0} "
            if 0 < q2 < 4:
                parts += f"{PAULI_STR[q2]}{g1} "
            
            # Erasure markers
            if e0:
                parts += f"X{g0 + eq_diff} "
            if e1:
                parts += f"X{g1 + eq_diff} "

            # Single join and append
            err_string.append(parts)

    return err_string
    
def hybrid_error_cnot(gates: List[tuple], p: float, f: float, eq_diff: int = 0):
    """Builds string list for hybrid Pauli + Erasure errors on CNOTs.
    
    All qubits can undergo errors. Each error has:
    - Fraction f: erasure error (heralded, Pauli + erasure marker)
    - Fraction 1-f: non-erasure error (unheralded, Pauli only)
    Total erasure probability per qubit is f*p.

    Args:
        gates: List of (control, target) qubit index tuples for CNOTs.
        p: Probability of error occurring on each CNOT.
        f: Fraction of errors which are erasures (0 <= f <= 1).
        eq_diff: Difference between data and fictitious ancilla qubit indices.

    Returns:
        List of strings representing the error operations to be appended to a Circuit._circ_str.
    """
    print(f"p: {p}, f: {f}")
    err_string = []
    errors = []
    PAULI_STR = ('I', 'X', 'Y', 'Z')
    
    else_space = 1.0

    for q1 in range(4):
            for q2 in range(4):
                for e1 in [0, 1]:
                    for e2 in [0, 1]:
                        if (q1, q2, e1, e2) == (0, 0, 0, 0): continue
                        
                        # For each qubit: 
                        # - No error (identity, no erasure): probability (1-p)
                        # - Pauli error without erasure: p*(1-f)/3 for each of X, Y, Z
                        # - Pauli error with erasure: p*f/4 for each of I, X, Y, Z
                        
                        if e1:
                            # Erasure on qubit 1
                            p1 = p * f / 4
                        else:
                            # No erasure on qubit 1
                            if q1 == 0:
                                p1 = 1 - p
                            else:
                                p1 = p * (1 - f) / 3

                        if e2:
                            # Erasure on qubit 2
                            p2 = p * f / 4
                        else:
                            # No erasure on qubit 2
                            if q2 == 0:
                                p2 = 1 - p
                            else:
                                p2 = p * (1 - f) / 3

                        true_prob = p1 * p2
                        errors.append((q1, q2, e1, e2, true_prob / else_space))
                        else_space -= true_prob

    # return errors

    for gate in gates:
        g0, g1 = gate[0], gate[1]
        
        for i in range(len(errors)):
            q1, q2, e0, e1, err_prob = errors[i]
            
            parts = ""
            
            # Error prefix
            if i == 0:
                parts += f"CORRELATED_ERROR({err_prob}) "
            else:
                parts += f"ELSE_CORRELATED_ERROR({err_prob}) "

            # Pauli errors (skip identity)
            if q1 != 0:
                parts += f"{PAULI_STR[q1]}{g0} "
            if q2 != 0:
                parts += f"{PAULI_STR[q2]}{g1} "
            
            # Erasure markers
            if e0:
                parts += f"X{g0 + eq_diff} "
            if e1:
                parts += f"X{g1 + eq_diff} "

            # Single join and append
            err_string.append(parts)

    return err_string

class Circuit:
    def __init__(self, code: codes.Stabilizer_Code):
        self._measurements = 0
        self._circ_str = []

        self.code = code

        self._hadamard_cache = {"meas_set": [],
                               "norm": None}

        self._observable_index = 0

        self._cached_strings = {}
        self._detector_cache = []

        self._measurement_sets = []
        self._measurement_sets_norm = []
        self._current_measurement_set = []

    def __repr__(self):
        return "\n".join(self._circ_str)
    
    def __str__(self):
        return "\n".join(self._circ_str)

    def set_noise_model(self, noise_model: noise.Noise_Model):
        self._noise_model = noise_model

    @property
    def noise_model(self):
        return self._noise_model

    @property
    def circ_str(self):
        return self._circ_str

    @property
    def cached_strings(self):
        return self._cached_strings

    @property
    def detector_cache(self):
        return self._detector_cache

    def append_to_circ_str(self, lines: List[str]):
        """(DEPRECATED) Appends multiple lines to the circuit string representation.
        
        Args:
            lines: List of strings to append.
        """
        self._circ_str.extend(lines)

    def add_reset(self, qubits: List[int], cache_name: str = None):
        """Adds reset operations for specified qubits.

        Args:
            qubits: List of qubit indices to reset.
        """
        self._circ_str.append("R " + " ".join(str(q) for q in qubits))

        if cache_name:
            self._cached_strings[cache_name] = self._circ_str[-1]

    def add_cnot(self, gates: List[Tuple[int, int]], cache_name: str):
        """Adds CNOT operations for specified gate pairs.
        
        Args:
            gates: List of (control, target) qubit index tuples for CNOTs.
        """
        self._circ_str.append("CX " + " ".join(f"{control} {target}" for control, target in gates))

        if cache_name:
            self._cached_strings[cache_name] = self._circ_str[-1]

    def add_depolarize1(self, qubits: List[int], p: float, cache_name: str = None):
        """Adds single-qubit depolarizing errors for specified qubits.

        Args:
            qubits: List of qubit indices to apply depolarizing errors.
            p: Probability of depolarizing error.
        """
        self._circ_str.append(f"DEPOLARIZE1({p}) " + " ".join(str(q) for q in qubits))

        if cache_name:
            self._cached_strings[cache_name] = self._circ_str[-1]

    def add_depolarize2(self, pairs: List[tuple], p: float, cache_name: str = None):
        """Adds two-qubit depolarizing errors for specified qubit pairs.

        Args:
            pairs: List of (q1, q2) qubit index tuples for depolarizing errors.
            p: Probability of depolarizing error.
        """
        self._circ_str.append(f"DEPOLARIZE2({p}) " + " ".join(f"{pair[0]} {pair[1]}" for pair in pairs))

        if cache_name:
            self._cached_strings[cache_name] = self._circ_str[-1]

    def add_erasure1(self, qubits: List[int], p: float, cache_name: str = None):
        """Adds single-qubit erasure errors for specified qubits.

        Args:
            qubits: List of qubit indices to apply erasure errors.
            p: Probability of erasure error.
        """
        measure_string = f"HERALDED_ERASE({p}) "
        num_qubits_measured = 0
        for q in qubits:
            measure_string += f"{q} "
            self._current_measurement_set.append(q)
            num_qubits_measured += 1

        self._circ_str.append(measure_string)
        self._measurements += num_qubits_measured
        self._measurement_sets.append(self._current_measurement_set)
        self._measurement_sets_norm.append(num_qubits_measured)
        self._current_measurement_set = []

        if cache_name:
            self._cached_strings[cache_name] = self._circ_str[-1]

    def add_symmetric_pauli_erasure_cnot(self, gates: List[tuple], erasure_bitmask: int, p: float, eq_diff: int=0):
        """Adds CNOT operations with symmetric Pauli + Erasure errors.

        Args:
            gates: List of (control, target) qubit index tuples for CNOTs.
            erasure_bitmask: Python int bitmask indicating which qubits have erasure errors. (WARNING: assumes all non-erasure qubits are Pauli qubits)
            p: Probability of error occurring on each CNOT.
            eq_diff: Difference between data and fictitious ancilla qubit indices.
        """
        self._circ_str.append("CX " + " ".join(f"{gate[0]} {gate[1]}" for gate in gates))
        if p > 0.0:
            self._circ_str.extend(pauli_erasure_cnot_optimized(gates, erasure_bitmask, p, eq_diff))

    def add_measurements(self, qubits: List[int], reset: bool = False, cache_name: str = None): # inefficient
        """Adds measurement operations for specified qubits.

        Args:
            qubits: List of qubit indices to measure.
        """
        measure_string = f"M{"R" if reset else ""} "
        num_qubits_measured = 0
        for q in qubits:
            measure_string += f"{q} "
            self._current_measurement_set.append(q)
            num_qubits_measured += 1

        self._circ_str.append(measure_string)
        self._measurements += num_qubits_measured
        self._measurement_sets.append(self._current_measurement_set)
        self._measurement_sets_norm.append(num_qubits_measured)
        self._current_measurement_set = []

        if cache_name:
            self._cached_strings[cache_name] = self._circ_str[-1]

    def add_h_gate(self, qubits: List[int], cache_name: str = None):
        """Adds Hadamard gates for specified qubits.

        Args:
            qubits: List of qubit indices to apply Hadamard gates.
            cache_name: Optional cache name for the operation.
        """
        self._circ_str.append(f"H " + " ".join(str(q) for q in qubits))

        if cache_name:
            self._cached_strings[cache_name] = self._circ_str[-1]

    def add_detectors(self, coords: List, parity: bool=False, cache: bool=False):
        """Adds detectors to the circuit.
        Args:
            coords: List of coordinates for detectors, or list of lists of coordinates for parity grouping.
            parity: If True, adds detectors with parity grouping.
        """
        if not parity:
            self._circ_str.append("\n".join([f"DETECTOR rec[-{coord}]" for coord in coords]))
            if cache:
                self.detector_cache.append(self._circ_str[-1])
        if parity:
            for coord in coords:
                self._circ_str.append(f"DETECTOR " + " ".join(f"rec[-{i}]" for i in coord))

                if cache:
                    self.detector_cache.append(self._circ_str[-1])  

    def detect_all_measurements(self):
        """Adds detectors for all measurements in the circuit, first-in first-out."""
        self.add_detectors(range(self._measurements, 0, -1))

    def add_observable(self, coords: List[int], cache: bool = False):
        """Adds an observable to the circuit.

        Args:
            coords: List of coordinates for the observable.
        """
        self._circ_str.append(f"OBSERVABLE_INCLUDE({self._observable_index}) " + " ".join(f"rec[-{coord}]" for coord in coords))
        
        if cache:
            self._cached_strings[f"obs_{self._observable_index}"] = self._circ_str[-1]

        self._observable_index += 1

    def get_measurement_sets(self):
        """Returns the measurement sets and their normalization factors.

        Returns:
            Tuple of (measurement_sets, measurement_sets_norm).
        """
        return self._measurement_sets, self._measurement_sets_norm

    def to_stim_circuit(self):
        """Converts the internal string representation to a stim.Circuit object."""
        return stim.Circuit("\n".join(self._circ_str))

def build_rsc_erasure_circuit(rsc: codes.RSC, noise_model: noise.Noise_Model, erasure_allocation_mode: Literal["all", "custom"]="all", custom_erasure_mask: int = None):
    """Builds the Stim circuit for the RSC code with specified noise.

    Args:
        code: The rotated surface code object
        noise_model: The noise model to apply
        erasure_allocation_mode: Mode for allocating erasure qubits ("all" or "custom")
        custom_erasure_mask: Custom erasure bitmask if erasure_allocation_mode is "custom"

    Returns:
        The associated Circuit object

    Raises:
        TypeError: If erasure_allocation_mode is invalid.
    """    

    circuit = Circuit(rsc)
    
    # TODO: cache traditional gates for later re-use
    if erasure_allocation_mode == "all" or (erasure_allocation_mode == "custom" and custom_erasure_mask is not None):
        circuit.pauli_bitmask, circuit.erasure_bitmask = rsc.allocate_erasure_qubits(erasure_allocation_mode, custom_erasure_mask)
    else:
        raise TypeError("Invalid erasure_allocation_mode. Choose from 'all', 'none', or 'custom' with a valid custom_erasure_mask.")

    circuit.set_noise_model(noise_model)
    
    # create bitmasks for erasure and pauli qubits
    noise_model.set_bitmasks(circuit.pauli_bitmask, circuit.erasure_bitmask)
    
    # define pauli/erasure support splits of various operations
    sp_pauli, sp_erasure, hadamard_pauli, hadamard_erasure, ancilla_meas_pauli, ancilla_meas_erasure = rsc._supports(circuit.pauli_bitmask)

    ## State preparation
    circuit.add_reset(rsc.all_qubit_ids, cache_name="sp")
    if sp_pauli and (p_sp_pauli := noise_model.noise_dict.get('sp', 0)) > 0:
        circuit.add_depolarize1(sp_pauli, p_sp_pauli, cache_name="sp_pauli")
    if sp_erasure and (p_sp_erasure := noise_model.noise_dict.get('sp-e', 0)) > 0:
        circuit.add_erasure1(sp_erasure, p_sp_erasure)

    ## X-ancilla Hadamards
    circuit.add_h_gate(rsc.x_ancilla_ids, cache_name="h")
    if hadamard_pauli and (p_sqg_pauli := noise_model.noise_dict.get('sqg', 0)) > 0:
        circuit.add_depolarize1(hadamard_pauli, p_sqg_pauli, cache_name="h_pauli")
    if hadamard_erasure and (p_sqg_erasure := noise_model.noise_dict.get('sqg-e', 0)) > 0:
        circuit.add_erasure1(hadamard_erasure, p_sqg_erasure)

    ## CNOT rounds
    fic_ancillas_bitmask = circuit.erasure_bitmask << rsc.eq_diff # generate fictitious ancillas for erasure handling
    for i, check_gates in enumerate(rsc.gates):
        cnot_bitmask = rsc.cnot_bitmasks[i]
        if (circuit.erasure_bitmask & cnot_bitmask >> rsc.eq_diff) > 0 and (p_tqg_e := noise_model.noise_dict.get('tqg-e', 0)) > 0:
            circuit.add_symmetric_pauli_erasure_cnot(check_gates, circuit.erasure_bitmask, p_tqg_e, rsc.eq_diff) # TODO: need to check if erasure bitmask hits cnot bitmask
            fic_ancillas_intersect_bitmask = fic_ancillas_bitmask & cnot_bitmask
            fic_ancillas_intersect = list(bitops.mask_iter_indices(fic_ancillas_intersect_bitmask))
            if (p_meas_pauli := noise_model.noise_dict.get('meas', 0)) > 0:
                circuit.add_depolarize1(fic_ancillas_intersect, p_meas_pauli)
            circuit.add_measurements(fic_ancillas_intersect, reset=True)
        elif (circuit.erasure_bitmask & cnot_bitmask >> rsc.eq_diff) == 0:
            circuit.add_cnot(check_gates, cache_name=f"cnot_{i}")
            circuit.add_depolarize2(check_gates, noise_model.noise_dict.get('tqg', 0), cache_name=f"cnot_{i}_pauli") # TODO: this process is inefficient (loops over check_gates twice)
        else:
            circuit.add_cnot(check_gates, cache_name=f"cnot_{i}")

    ## X-ancilla Hadamards. TODO: Figure out a way to cache these properly
    circuit.add_h_gate(rsc.x_ancilla_ids)
    if hadamard_pauli and (p_sqg_pauli := noise_model.noise_dict.get('sqg', 0)) > 0:
        circuit.add_depolarize1(hadamard_pauli, p_sqg_pauli)
    if hadamard_erasure and (p_sqg_erasure := noise_model.noise_dict.get('sqg-e', 0)) > 0:
        circuit.add_erasure1(hadamard_erasure, p_sqg_erasure)

    ## Add detectors for all measurements so far
    circuit.detect_all_measurements()

    ## Measure all ancillas
    circuit.add_measurements(rsc.x_ancilla_ids + rsc.z_ancilla_ids, cache_name="meas")
    # ancilla_meas_pauli, ancilla_meas_erasure = split_support_list_fast(self.ancilla_ids, pauli_qubits, erasure_qubits)
    if ancilla_meas_pauli and (p_anc_meas_pauli := noise_model.noise_dict.get('meas', 0)) > 0:
        circuit.add_depolarize1(ancilla_meas_pauli, p_anc_meas_pauli, cache_name="meas_pauli")
    if ancilla_meas_erasure and (p_anc_meas_erasure := noise_model.noise_dict.get('meas-e', 0)) > 0:
        circuit.add_erasure1(ancilla_meas_erasure, p_anc_meas_erasure)

    ## Z-ancilla detectors
    circuit.add_detectors(range(rsc.num_ancillas // 2, 0, -1), cache=True)

    ## Measure all data qubits
    circuit.add_measurements(rsc.data_qubit_ids, cache_name="dmeas")
    # TODO: Add measurement noise

    ## Measure plaquettes
    circuit.add_detectors([[rsc.num_qubits - rsc.measurement_indices[anc_id] for anc_id in rsc.plaquettes[z_anc_id] + [z_anc_id]] for z_anc_id in rsc.z_ancilla_ids], parity=True, cache=True)

    ## Observable
    circuit.add_observable([rsc.num_qubits - rsc.measurement_indices[qubit_id] for qubit_id in rsc.observable], cache=True)

    return circuit

def build_rsc_hybrid_circuit(rsc: codes.RSC, noise_model: noise.Noise_Model):
    """Builds the Stim circuit for the RSC code with specified noise. WARNING: Currently only supports hybrid Pauli-erasure noise.

    Args:
        code: The rotated surface code object
        noise_model: The noise model to apply
    
    Returns:
        The associated Circuit object
    """
    
    circuit = Circuit(rsc)
    circuit.set_noise_model(noise_model)
    
    # For hybrid circuits, all qubits can have erasures
    # Set bitmasks for batch builder compatibility
    all_qubit_mask = bitops.indices_to_mask(rsc.all_qubit_ids)
    circuit.erasure_bitmask = all_qubit_mask
    circuit.pauli_bitmask = all_qubit_mask

    ## State preparation
    circuit.add_reset(rsc.all_qubit_ids, cache_name="sp")
    # TODO: Add state prep noise support

    ## X-ancilla Hadamards
    circuit.add_h_gate(rsc.x_ancilla_ids, cache_name="h")
    # TODO: Add single-qubit gate noise support

    ## CNOT rounds
    for i, check_gates in enumerate(rsc.gates):
        circuit.add_cnot(check_gates, cache_name=f"cnot_{i}")
        if (p_tqg := noise_model.noise_dict.get('p', 0)) > 0 and (f_tqg := noise_model.noise_dict.get('f', 0)) > 0:
            circuit._circ_str.extend(hybrid_error_cnot(gates=check_gates, p=p_tqg, f=f_tqg, eq_diff=rsc.eq_diff))
        elif (p_tqg := noise_model.noise_dict.get('p', 0)) > 0:
            circuit.add_depolarize1([q for gate in check_gates for q in gate], p_tqg)
        
        circuit.add_measurements([q + rsc.eq_diff for gate in check_gates for q in gate], reset=True)

    ## X-ancilla Hadamards
    circuit.add_h_gate(rsc.x_ancilla_ids)
    # TODO: Add single-qubit gate noise support

    ## Detect all measurements
    circuit.detect_all_measurements()

    ## Measure all ancillas
    circuit.add_measurements(rsc.x_ancilla_ids + rsc.z_ancilla_ids, cache_name="meas")
    circuit.add_detectors(range(rsc.num_ancillas // 2, 0, -1), cache=True)

    ## Measure all data qubits
    circuit.add_measurements(rsc.data_qubit_ids, cache_name="dmeas")

    ## Measure plaquettes
    circuit.add_detectors([[rsc.num_qubits - rsc.measurement_indices[anc_id] for anc_id in rsc.plaquettes[z_anc_id] + [z_anc_id]] for z_anc_id in rsc.z_ancilla_ids], parity=True, cache=True)

    ## Observable
    circuit.add_observable([rsc.num_qubits - rsc.measurement_indices[qubit_id] for qubit_id in rsc.observable], cache=True)

    return circuit

def build_repetition_code_hybrid_circuit(rep: codes.RepetitionCode, noise_model: noise.Noise_Model):
    """Builds the Stim circuit for the Repetition code with specified noise. WARNING: Currently only supports hybrid Pauli-erasure noise.

    Args:
        code: The repetition code object
        noise_model: The noise model to apply

    Returns:
        The associated Circuit object
    """
    
    circuit = Circuit(rep)
    circuit.set_noise_model(noise_model)

    # TODO: bitmask operations

    ## State preparation
    circuit.add_reset(rep.all_qubit_ids, cache_name="sp")
    # TODO: Add state prep noise support

    ## CNOT rounds
    for i in range(2):
        circuit.add_cnot(rep.gates[i], cache_name=f"cnot_{i}")
        if (p_tqg := noise_model.noise_dict.get('p', 0)) > 0 and (f_tqg := noise_model.noise_dict.get('f', 0)) > 0:
            circuit._circ_str.extend(hybrid_error_cnot(gates=rep.gates[i], p=p_tqg, f=f_tqg, eq_diff=rep.eq_diff))
        elif (p_tqg := noise_model.noise_dict.get('p', 0)) > 0:
            circuit.add_depolarize1([q for gate in rep.gates[i] for q in gate], p_tqg)
        
        circuit.add_measurements([q + rep.eq_diff for gate in rep.gates[i] for q in gate], reset=True)

    ## Detect all measurements
    circuit.detect_all_measurements()

    ## Measure all ancillas
    circuit.add_measurements(rep.ancilla_ids, cache_name="meas")
    circuit.add_detectors(range(rep.num_ancillas, 0, -1), cache=True)

    ## Measure all data qubits
    circuit.add_measurements(rep.data_qubit_ids, cache_name="dmeas")

    ## Measure plaquettes
    circuit.add_detectors([[rep.num_qubits - rep.measurement_indices[anc_id] for anc_id in rep.plaquettes[z_anc_id] + [z_anc_id]] for z_anc_id in rep.z_ancilla_ids], parity=True, cache=True)

    ## Observable
    circuit.add_observable([1], cache=True)

    return circuit

def cliffordize_hybrid(hybrid_circuit: Circuit, samples: np.ndarray) -> List[Circuit]:
    """Converts a hybrid error circuit to a Clifford circuit by replacing erasure errors with depolarizing errors.

    Args:
        hybrid_circuit: The original Circuit object with hybrid errors.
        samples: Array of erasure syndrome samples (shape: num_samples × num_erasure_measurements)

    Returns:
        List of Circuit objects with erasure errors replaced by depolarizing errors.
    """
    circuit_cache = {}
    circuits = []
    
    # Determine if RSC or RepetitionCode
    is_rsc = isinstance(hybrid_circuit.code, codes.RSC)
    is_rep = isinstance(hybrid_circuit.code, codes.RepetitionCode)
    
    if not (is_rsc or is_rep):
        raise NotImplementedError("Currently only RepetitionCode and RSC are supported for cliffordization of hybrid circuits.")

    for sample in samples:
        sample_bytes = sample.tobytes()
        if sample_bytes in circuit_cache:
            circuits.append(circuit_cache[sample_bytes])
            continue

        circuit = Circuit(hybrid_circuit.code)
        circuit.set_noise_model(hybrid_circuit.noise_model)

        ## State preparation
        circuit._circ_str.append(hybrid_circuit.cached_strings["sp"])
        
        if is_rsc:
            ## X-ancilla Hadamards
            circuit._circ_str.append(hybrid_circuit.cached_strings["h"])
        
        ## CNOT rounds
        sample_index = 0
        num_rounds = 4 if is_rsc else 2
        
        for gate_index in range(num_rounds):
            cnot_gates = hybrid_circuit.code.gates[gate_index]
            circuit._circ_str.append(hybrid_circuit.cached_strings[f"cnot_{gate_index}"])
            erased = []
            unerased = []

            # For each CNOT gate, check if either qubit was erased
            for gate in cnot_gates:
                control, target = gate
                # Check erasure measurements for both qubits
                control_erased = sample[sample_index] if sample_index < len(sample) else False
                target_erased = sample[sample_index + 1] if sample_index + 1 < len(sample) else False
                
                if control_erased:
                    erased.append(control)
                else:
                    unerased.append(control)
                
                if target_erased:
                    erased.append(target)
                else:
                    unerased.append(target)
                
                sample_index += 2

            # Remove duplicates
            erased = list(set(erased))
            unerased = list(set(unerased))

            # Apply depolarizing noise based on whether qubits were erased
            if (p := hybrid_circuit.noise_model.noise_dict.get('p', 0)) > 0:
                f = hybrid_circuit.noise_model.noise_dict.get('f', 0)
                if unerased:
                    # Non-erased qubits get depolarizing noise with rate p*(1-f)
                    circuit.add_depolarize1(unerased, p * (1 - f))
                
                if erased:
                    # Erased qubits get depolarizing noise with rate 0.75 (maximally mixed)
                    circuit.add_depolarize1(erased, 0.75)
        
        if is_rsc:
            ## X-ancilla Hadamards
            circuit.add_h_gate(hybrid_circuit.code.x_ancilla_ids)
        
        ## Measure all ancillas
        circuit._circ_str.append(hybrid_circuit.cached_strings["meas"])
        
        if is_rsc:
            # Z-ancilla detectors (skip detector_cache[0] which is for erasure measurements)
            circuit._circ_str.append(hybrid_circuit.detector_cache[0])
        else:
            # RepetitionCode detectors (skip detector_cache[0] which is for erasure measurements)
            circuit.append_to_circ_str(hybrid_circuit.detector_cache[1: 1 + hybrid_circuit.code.num_ancillas])
        
        ## Measure all data qubits
        circuit._circ_str.append(hybrid_circuit.cached_strings["dmeas"])
        
        ## Plaquette detectors
        if is_rsc:
            circuit.append_to_circ_str(hybrid_circuit.detector_cache[1:])
        else:
            circuit.append_to_circ_str(hybrid_circuit.detector_cache[1 + hybrid_circuit.code.num_ancillas:])
        
        ## Observable
        circuit._circ_str.append(hybrid_circuit.cached_strings["obs_0"])

        circuit_cache[sample_bytes] = circuit
        circuits.append(circuit)

    return circuits
