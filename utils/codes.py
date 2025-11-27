import numpy as np
import matplotlib.pyplot as plt
import stim
import time

from typing import Iterable, Optional, Tuple, Literal
from stimdecoders.utils import bitops, noise

# TODO: Docstrins
class _Qubit:
    """
    Utility class representing a qubit

    Args:
        type (str): Type of qubit ('data', 'x-ancilla', 'z-ancilla')
        coords (tuple): Coordinates of the qubit in the lattice
        id (int, optional): Unique identifier for the qubit. Defaults to None.
    """
    def __init__(self, type: str, coords: tuple, id: int = None):
        self._id = id
        self._coords = coords
        self._type = type

    def __repr__(self):
        return f"Qubit ID: {self.id}, Type: {self.type}, Coords: {self.coords}"

    @property
    def type(self):
        return self._type

    @property
    def coords(self):
        return self._coords
    
    @property
    def id(self):
        return self._id

# TODO: Docstrings
class Stabilizer_Code():
    def __init__(self, distance: int, check_density: int):
        self.distance = distance
        self.gates = [[] for _ in range(check_density)]

        self.observable = []
        self.plaquettes = {}

        self.data_qubits = {}
        self.x_ancillas = {}
        self.z_ancillas = {}

        self.all_qubit_ids = []
        self.data_qubit_ids = []
        self.ancilla_ids = []
        self.x_ancilla_ids = []
        self.z_ancilla_ids = []

        self.qubit_ids = {}

    def add_data_qubit(self, coord: tuple, id: int):
        self.data_qubits[coord] = _Qubit(type='data', coords=coord, id=id)
        self.qubit_ids[id] = self.data_qubits[coord]
        self.all_qubit_ids.append(id)
        self.data_qubit_ids.append(id)

    def add_x_ancilla(self, coord: tuple, id: int):
        self.x_ancillas[coord] = _Qubit(type='x-ancilla', coords=coord, id=id)
        self.qubit_ids[id] = self.x_ancillas[coord]
        self.plaquettes[id] = []
        self.all_qubit_ids.append(id)
        self.ancilla_ids.append(id)
        self.x_ancilla_ids.append(id)

    def add_z_ancilla(self, coord: tuple, id: int):
        self.z_ancillas[coord] = _Qubit(type='z-ancilla', coords=coord, id=id)
        self.qubit_ids[id] = self.z_ancillas[coord]
        self.plaquettes[id] = []
        self.all_qubit_ids.append(id)
        self.ancilla_ids.append(id)
        self.z_ancilla_ids.append(id)

    def qubit_with_id(self, id: int) -> _Qubit:
        return self.qubit_ids.get(id)

class RSC(Stabilizer_Code):
    def __init__(self, distance: int):
        super().__init__(distance, check_density=4)
        self.num_ancillas = distance**2 - 1
        self.num_qubits = 2*(distance**2) - 1
        self.measurement_indices = {}

        self.sp_support = 0b0
        self.hadamard_support = 0b0
        self.ancilla_measure_support = 0b0

        self.cached_strings = []

        self._build_lattice()
        self._build_checks()

    def allocate_erasure_qubits(self, mode: Literal["all", "none", "custom"], custom_erasure_mask: Optional[int] = None):
        """Allocates erasure qubits based on the specified mode.

        Args:
            mode (Literal["all", "none", "custom"]): Mode of erasure allocation.
                "all": All qubits are erasure qubits.
                "none": No qubits are erasure qubits.
                "custom": Use a custom erasure mask.
            custom_erasure_mask (Optional[int], optional): Custom erasure bitmask. Required if mode is "custom". Defaults to None.

        Raises:
            ValueError: If mode is "custom" and custom_erasure_mask is not provided.
        """
        all_qubit_mask = bitops.indices_to_mask(self.all_qubit_ids)
        if mode == "all":
            erasure_bitmask = all_qubit_mask
            pauli_bitmask = 0b0
        elif mode == "none":
            erasure_bitmask = 0b0
            pauli_bitmask = all_qubit_mask
        elif mode == "custom":
            if custom_erasure_mask is None:
                raise ValueError("Custom erasure mask must be provided when mode is 'custom'.")
            self.erasure_bitmask = custom_erasure_mask
            self.pauli_bitmask = (~custom_erasure_mask) & all_qubit_mask
        else:
            raise ValueError("Invalid mode. Choose from 'all', 'none', or 'custom'.")

        return pauli_bitmask, erasure_bitmask

    def _build_lattice(self):
        """Builds the rotated surface code lattice."""
        x_index = 0
        z_index = self.num_ancillas // 2
        data_index = self.num_ancillas

        for index in range((2*self.distance + 1)*(self.distance + 1)):
            x = index % (2*self.distance + 1)
            y = 2*(index // (2*self.distance + 1)) + (x % 2)

            if x % 2 == 1 and y % 2 == 1 and x < 2*self.distance and y < 2*self.distance:
                self.add_data_qubit((x, y), index)
                self.measurement_indices[index] = data_index
                self.sp_support |= (1 << index)
                data_index += 1
                if y == 1:
                    self.observable.append(index)
            elif (x + y) % 4 != 0 and x > 1 and x < (2*self.distance - 1) and y < (2*self.distance + 1):
                self.add_x_ancilla((x, y), index)
                self.measurement_indices[index] = x_index
                self.hadamard_support |= (1 << index)
                self.ancilla_measure_support |= (1 << index)
                x_index += 1
            elif (x + y) % 4 == 0 and x < (2*self.distance + 1) and y > 1 and y < (2*self.distance - 1):
                self.add_z_ancilla((x, y), index)
                self.measurement_indices[index] = z_index
                self.ancilla_measure_support |= (1 << index)
                z_index += 1

        self.eq_diff = self.x_ancilla_ids[-1]

    def _build_checks(self):
        """Builds the CNOT gates for each check of the RSC code."""
        self.cnot_bitmasks = [0b0, 0b0, 0b0, 0b0]
        x_order = [(-1, 1), (1, 1), (-1, -1), (1, -1)]
        z_order = [(-1, 1), (-1, -1), (1, 1), (1, -1)]
        for check_num in range(4):
            for x_ancilla in self.x_ancillas.values():
                if qubit := self.data_qubits.get((x_ancilla.coords[0] + x_order[check_num][0], x_ancilla.coords[1] + x_order[check_num][1])):
                    self.gates[check_num].append((x_ancilla.id, qubit.id))
                    # self.plaquettes[x_ancilla.id].append(qubit.id) # unnecessary for code initialization in Z eigenstate
                    self.cnot_bitmasks[check_num] |= (((1 << qubit.id) | (1 << x_ancilla.id)) << self.eq_diff)

            for z_ancilla in self.z_ancillas.values():
                if qubit := self.data_qubits.get((z_ancilla.coords[0] + z_order[check_num][0], z_ancilla.coords[1] + z_order[check_num][1])):
                    self.gates[check_num].append((qubit.id, z_ancilla.id))
                    self.plaquettes[z_ancilla.id].append(qubit.id)
                    self.cnot_bitmasks[check_num] |= (((1 << qubit.id) | (1 << z_ancilla.id)) << self.eq_diff)

    def _supports(self, pauli_bitmask):
        """Determines support lists for various operations based on error bitmasks.
        
        Args:
            pauli_bitmask: Integer bitmask for Pauli errors. (WARNING: assumes that all non-Pauli qubits are Erasures)
        
        Returns:
            Tuple[List[int], List[int], List[int], List[int], List[int], List[int]]: Support lists for the following error operations: sp_pauli
                sp_erasure, hadamard_pauli, hadamard_erasure, ancilla_meas_pauli, ancilla_meas_erasure
        """
        sp_pauli = []
        sp_erasure = []
        hadamard_pauli = []
        hadamard_erasure = []
        ancilla_meas_pauli = []
        ancilla_meas_erasure = []

        for i in self.all_qubit_ids:
            if (pauli_bitmask >> i) & 1:
                if (self.sp_support >> i) & 1: 
                    sp_pauli.append(i)
                    continue

                ancilla_meas_pauli.append(i)
                if (self.hadamard_support >> i) & 1: hadamard_pauli.append(i)
            else:
                if (self.sp_support >> i) & 1: 
                    sp_erasure.append(i)
                    continue

                ancilla_meas_erasure.append(i)
                if (self.hadamard_support >> i) & 1: hadamard_erasure.append(i)

        return sp_pauli, sp_erasure, hadamard_pauli, hadamard_erasure, ancilla_meas_pauli, ancilla_meas_erasure

    def draw_lattice(self, numbering=True):
        """
        Draws the lattice of the RSC code as an SVG with grid.

        Args:
            numbering (bool): Whether to number the qubits with their IDs.

        Returns:
            IPython.display.SVG or str: SVG display object in Jupyter, or SVG string otherwise.
        """
        # Find bounds
        all_coords = list(self.data_qubits.keys()) + list(self.x_ancillas.keys()) + list(self.z_ancillas.keys())
        if not all_coords:
            svg_string = '<svg width="100" height="100"></svg>'
        else:
            # Find max ID length to size circles appropriately
            all_ids = [q.id for q in self.data_qubits.values()] + \
                      [q.id for q in self.x_ancillas.values()] + \
                      [q.id for q in self.z_ancillas.values()]
            max_id_digits = len(str(max(all_ids))) if all_ids else 1
            
            # Adaptive radius based on number of digits
            if max_id_digits == 1:
                radius = 10
                font_size = 10
            elif max_id_digits == 2:
                radius = 12
                font_size = 9
            else:  # 3+ digits
                radius = 13
                font_size = 8
            
            min_x = min(c[0] for c in all_coords)
            max_x = max(c[0] for c in all_coords)
            min_y = min(c[1] for c in all_coords)
            max_y = max(c[1] for c in all_coords)
            
            # Scale and padding
            scale = 30  # pixels per unit
            padding = 40
            
            width = int((max_x - min_x) * scale + 2 * padding)
            height = int((max_y - min_y) * scale + 2 * padding)
            
            def transform_x(x):
                return (x - min_x) * scale + padding
            
            def transform_y(y):
                return (y - min_y) * scale + padding
            
            svg_parts = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
            
            # Draw grid (light gray lines) - only through data qubits
            # Get unique x and y coordinates of data qubits
            data_x_coords = sorted(set(coord[0] for coord in self.data_qubits.keys()))
            data_y_coords = sorted(set(coord[1] for coord in self.data_qubits.keys()))
            
            # Vertical lines through data qubits
            for x in data_x_coords:
                x_pos = transform_x(x)
                y_start = transform_y(min(data_y_coords))
                y_end = transform_y(max(data_y_coords))
                svg_parts.append(f'<line x1="{x_pos}" y1="{y_start}" x2="{x_pos}" y2="{y_end}" stroke="#e0e0e0" stroke-width="3"/>')
            
            # Horizontal lines through data qubits
            for y in data_y_coords:
                y_pos = transform_y(y)
                x_start = transform_x(min(data_x_coords))
                x_end = transform_x(max(data_x_coords))
                svg_parts.append(f'<line x1="{x_start}" y1="{y_pos}" x2="{x_end}" y2="{y_pos}" stroke="#e0e0e0" stroke-width="3"/>')
            
            # Draw data qubits (black)
            for coord, qubit in self.data_qubits.items():
                x, y = transform_x(coord[0]), transform_y(coord[1])
                svg_parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="black"/>')
                if numbering:
                    svg_parts.append(f'<text x="{x}" y="{y + font_size//3}" font-size="{font_size}" font-family="monospace" fill="white" text-anchor="middle">{qubit.id}</text>')
            
            # Draw x-ancillas (soft red)
            for coord, qubit in self.x_ancillas.items():
                x, y = transform_x(coord[0]), transform_y(coord[1])
                svg_parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="#ff6b6b"/>')
                if numbering:
                    svg_parts.append(f'<text x="{x}" y="{y + font_size//3}" font-size="{font_size}" font-family="monospace" fill="white" text-anchor="middle">{qubit.id}</text>')
            
            # Draw z-ancillas (soft blue)
            for coord, qubit in self.z_ancillas.items():
                x, y = transform_x(coord[0]), transform_y(coord[1])
                svg_parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="#4dabf7"/>')
                if numbering:
                    svg_parts.append(f'<text x="{x}" y="{y + font_size//3}" font-size="{font_size}" font-family="monospace" fill="white" text-anchor="middle">{qubit.id}</text>')
            
            svg_parts.append('</svg>')
            svg_string = '\n'.join(svg_parts)
        
        # Return IPython SVG object if in Jupyter, otherwise return string
        try:
            from IPython.display import SVG
            return SVG(svg_string)
        except ImportError:
            return svg_string

    def draw_checks(self):
        """
        Draws the CNOT gates for each check of the RSC code as SVG diagrams.

        Returns:
            IPython.display.SVG or str: SVG display object showing all 4 checks.
        """
        # Find max ID length for sizing
        all_ids = [q.id for q in self.data_qubits.values()] + \
                  [q.id for q in self.x_ancillas.values()] + \
                  [q.id for q in self.z_ancillas.values()]
        max_id_digits = len(str(max(all_ids))) if all_ids else 1
        
        # Adaptive radius based on number of digits
        if max_id_digits == 1:
            radius = 10
            font_size = 10
        elif max_id_digits == 2:
            radius = 12
            font_size = 9
        else:  # 3+ digits
            radius = 15
            font_size = 8
        
        # Get bounds for each check
        all_coords = list(self.data_qubits.keys()) + list(self.x_ancillas.keys()) + list(self.z_ancillas.keys())
        min_x = min(c[0] for c in all_coords)
        max_x = max(c[0] for c in all_coords)
        min_y = min(c[1] for c in all_coords)
        max_y = max(c[1] for c in all_coords)
        
        scale = 30
        padding = 50  # Extra padding for labels
        single_width = int((max_x - min_x) * scale + 2 * padding)
        single_height = int((max_y - min_y) * scale + 2 * padding)
        
        # Total SVG size (2x2 grid) - reduced spacing between diagrams
        spacing = 10  # Reduced from 20
        total_width = single_width * 2 + spacing * 3
        total_height = single_height * 2 + spacing * 3
        
        def transform_x(x, offset_x=0):
            return (x - min_x) * scale + padding + offset_x
        
        def transform_y(y, offset_y=0):
            return (y - min_y) * scale + padding + offset_y
        
        svg_parts = [f'<svg width="{total_width}" height="{total_height}" xmlns="http://www.w3.org/2000/svg">']
        
        # Draw 4 checks in 2x2 grid
        for check_num in range(4):
            row = check_num // 2
            col = check_num % 2
            offset_x = col * (single_width + spacing) + spacing
            offset_y = row * (single_height + spacing) + spacing
            
            # Title
            svg_parts.append(f'<text x="{offset_x + single_width//2}" y="{offset_y + 20}" font-size="14" font-weight="bold" font-family="monospace" text-anchor="middle">CHECK {check_num + 1}</text>')

            # Draw grid through data qubits
            data_x_coords = sorted(set(coord[0] for coord in self.data_qubits.keys()))
            data_y_coords = sorted(set(coord[1] for coord in self.data_qubits.keys()))
            
            for x in data_x_coords:
                x_pos = transform_x(x, offset_x)
                y_start = transform_y(min(data_y_coords), offset_y)
                y_end = transform_y(max(data_y_coords), offset_y)
                svg_parts.append(f'<line x1="{x_pos}" y1="{y_start}" x2="{x_pos}" y2="{y_end}" stroke="#e0e0e0" stroke-width="3"/>')
            
            for y in data_y_coords:
                y_pos = transform_y(y, offset_y)
                x_start = transform_x(min(data_x_coords), offset_x)
                x_end = transform_x(max(data_x_coords), offset_x)
                svg_parts.append(f'<line x1="{x_start}" y1="{y_pos}" x2="{x_end}" y2="{y_pos}" stroke="#e0e0e0" stroke-width="3"/>')
            
            # Collect qubits involved in CNOTs for this check
            cnot_qubits = set()
            for gate in self.gates[check_num]:
                cnot_qubits.add(gate[0])
                cnot_qubits.add(gate[1])
            
            # Draw all qubits (those NOT in CNOTs)
            for coord, qubit in self.data_qubits.items():
                if qubit.id not in cnot_qubits:
                    x, y = transform_x(coord[0], offset_x), transform_y(coord[1], offset_y)
                    svg_parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="black"/>')
                    svg_parts.append(f'<text x="{x}" y="{y + font_size//3}" font-size="{font_size}" font-family="monospace" fill="white" text-anchor="middle">{qubit.id}</text>')
            
            # for coord, qubit in self.x_ancillas.items():
            #     if qubit.id not in cnot_qubits:
            #         x, y = transform_x(coord[0], offset_x), transform_y(coord[1], offset_y)
            #         svg_parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="#ff6b6b"/>')
            #         svg_parts.append(f'<text x="{x}" y="{y + font_size//3}" font-size="{font_size}" fill="white" text-anchor="middle">{qubit.id}</text>')
            
            for coord, qubit in self.x_ancillas.items():
                if qubit.id not in cnot_qubits:
                    x, y = transform_x(coord[0], offset_x), transform_y(coord[1], offset_y)
                    svg_parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="#ff6b6b"/>')
                    svg_parts.append(f'<text x="{x}" y="{y + font_size//3}" font-size="{font_size}" font-family="monospace" fill="white" text-anchor="middle">{qubit.id}</text>')
            
            for coord, qubit in self.z_ancillas.items():
                if qubit.id not in cnot_qubits:
                    x, y = transform_x(coord[0], offset_x), transform_y(coord[1], offset_y)
                    svg_parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="#4dabf7"/>')
                    svg_parts.append(f'<text x="{x}" y="{y + font_size//3}" font-size="{font_size}" font-family="monospace" fill="white" text-anchor="middle">{qubit.id}</text>')
            
            # Draw CNOT gates
            for gate in self.gates[check_num]:
                control = self.qubit_with_id(gate[0])
                target = self.qubit_with_id(gate[1])
                
                # Color based on ancilla type
                color = '#ff6b6b' if control.type == 'x-ancilla' else '#4dabf7'
                
                cx = transform_x(control.coords[0], offset_x)
                cy = transform_y(control.coords[1], offset_y)
                tx = transform_x(target.coords[0], offset_x)
                ty = transform_y(target.coords[1], offset_y)
                
                # Calculate line endpoints to stop at target edge
                target_radius = 8
                # Calculate direction vector and normalize
                dx = tx - cx
                dy = ty - cy
                length = (dx**2 + dy**2)**0.5
                if length > 0:
                    dx_norm = dx / length
                    dy_norm = dy / length
                    # Stop at control edge (5px radius) and target edge (8px radius)
                    line_start_x = cx + dx_norm * 5
                    line_start_y = cy + dy_norm * 5
                    line_end_x = tx - dx_norm * target_radius
                    line_end_y = ty - dy_norm * target_radius
                else:
                    line_start_x, line_start_y = cx, cy
                    line_end_x, line_end_y = tx, ty
                
                # Draw vertical line connecting control and target (stops at target edge)
                svg_parts.append(f'<line x1="{line_start_x}" y1="{line_start_y}" x2="{line_end_x}" y2="{line_end_y}" stroke="{color}" stroke-width="2"/>')
                
                # Draw control (filled circle)
                svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="{color}"/>')
                
                # Draw target (circle with plus sign)
                svg_parts.append(f'<circle cx="{tx}" cy="{ty}" r="{target_radius}" fill="white" stroke="{color}" stroke-width="2"/>')
                svg_parts.append(f'<line x1="{tx}" y1="{ty - target_radius}" x2="{tx}" y2="{ty + target_radius}" stroke="{color}" stroke-width="2"/>')
                svg_parts.append(f'<line x1="{tx - target_radius}" y1="{ty}" x2="{tx + target_radius}" y2="{ty}" stroke="{color}" stroke-width="2"/>')
                
                # Adaptive label positioning - place closer to gate, offset based on direction
                # Determine which side to place label (prefer left, but adapt if line is nearly vertical)
                label_offset = 12  # Closer to the gate
                
                # For control qubit
                if abs(dx) > abs(dy):  # More horizontal
                    c_label_x = cx - label_offset if dx > 0 else cx + label_offset
                    c_label_anchor = "end" if dx > 0 else "start"
                    c_label_y = cy + 4
                else:  # More vertical
                    c_label_x = cx - label_offset
                    c_label_anchor = "end"
                    c_label_y = cy + 4
                
                # For target qubit
                if abs(dx) > abs(dy):  # More horizontal
                    t_label_x = tx - label_offset if dx < 0 else tx + label_offset
                    t_label_anchor = "end" if dx < 0 else "start"
                    t_label_y = ty + 4
                else:  # More vertical
                    t_label_x = tx - label_offset
                    t_label_anchor = "end"
                    t_label_y = ty + 4
                
                # Draw qubit IDs with adaptive positioning
                svg_parts.append(f'<text x="{c_label_x}" y="{c_label_y}" font-size="10" font-family="monospace" fill="black" text-anchor="{c_label_anchor}">{gate[0]}</text>')
                svg_parts.append(f'<text x="{t_label_x}" y="{t_label_y}" font-size="10" font-family="monospace" fill="black" text-anchor="{t_label_anchor}">{gate[1]}</text>')
        
        svg_parts.append('</svg>')
        svg_string = '\n'.join(svg_parts)
        
        # Return IPython SVG object if in Jupyter, otherwise return string
        try:
            from IPython.display import SVG
            return SVG(svg_string)
        except ImportError:
            return svg_string
    
class RepetitionCode(Stabilizer_Code):
    def __init__(self, distance: int):
        super().__init__(distance, check_density=2)
        del self.x_ancillas
        del self.x_ancilla_ids

        self.num_qubits = 2*distance - 1
        self.num_data_qubits = distance
        self.num_ancillas = distance - 1

        self.measurement_indices = {}

        self._build_lattice()
        self._build_checks()

    def _build_lattice(self):
        """Builds the lattice of the repetition code."""
        index = 1
        ancilla_index = 0
        data_index = self.num_ancillas

        for i in range(self.distance):
            self.add_data_qubit((2*i, 0), index)
            self.observable.append(index)
            self.measurement_indices[index] = data_index
            data_index += 1
            index += 1

            if i < self.distance - 1:
                self.add_z_ancilla((2*i + 1, 1), index)
                self.measurement_indices[index] = ancilla_index
                ancilla_index += 1
                index += 1

        self.eq_diff = index

    def _build_checks(self):
        """Builds the checks for the repetition code."""
        for i in range(self.distance - 1):
            ancilla_id = self.z_ancilla_ids[i]
            self.gates[0].append((ancilla_id - 1, ancilla_id))
            self.gates[1].append((ancilla_id + 1, ancilla_id))
            self.plaquettes[ancilla_id].extend([ancilla_id - 1, ancilla_id + 1])

    def draw_lattice(self, numbering=True):
        """
        Draws the lattice of the repetition code as an SVG.

        Args:
            numbering (bool): Whether to number the qubits with their IDs.

        Returns:
            IPython.display.SVG or str: SVG display object in Jupyter, or SVG string otherwise.
        """
        # Find bounds
        all_coords = list(self.data_qubits.keys()) + list(self.z_ancillas.keys())
        if not all_coords:
            svg_string = '<svg width="100" height="100"></svg>'
        else:
            # Find max ID length to size circles appropriately
            all_ids = [q.id for q in self.data_qubits.values()] + \
                      [q.id for q in self.z_ancillas.values()]
            max_id_digits = len(str(max(all_ids))) if all_ids else 1
            
            # Adaptive radius based on number of digits
            if max_id_digits == 1:
                radius = 10
                font_size = 10
            elif max_id_digits == 2:
                radius = 12
                font_size = 9
            else:  # 3+ digits
                radius = 13
                font_size = 8
            
            min_x = min(c[0] for c in all_coords)
            max_x = max(c[0] for c in all_coords)
            min_y = min(c[1] for c in all_coords)
            max_y = max(c[1] for c in all_coords)
            
            # Scale and padding
            scale = 30  # pixels per unit
            padding = 40
            
            width = int((max_x - min_x) * scale + 2 * padding)
            height = int((max_y - min_y) * scale + 2 * padding)
            
            def transform_x(x):
                return (x - min_x) * scale + padding
            
            def transform_y(y):
                return (y - min_y) * scale + padding
            
            svg_parts = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
            
            # Draw horizontal line through data qubits
            data_coords = sorted(self.data_qubits.keys())
            if len(data_coords) >= 2:
                x_start = transform_x(data_coords[0][0])
                x_end = transform_x(data_coords[-1][0])
                y_line = transform_y(data_coords[0][1])
                svg_parts.append(f'<line x1="{x_start}" y1="{y_line}" x2="{x_end}" y2="{y_line}" stroke="#e0e0e0" stroke-width="3"/>')
            
            # Draw data qubits (black)
            for coord, qubit in self.data_qubits.items():
                x, y = transform_x(coord[0]), transform_y(coord[1])
                svg_parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="black"/>')
                if numbering:
                    svg_parts.append(f'<text x="{x}" y="{y + font_size//3}" font-size="{font_size}" font-family="monospace" fill="white" text-anchor="middle">{qubit.id}</text>')
            
            # Draw z-ancillas (soft blue)
            for coord, qubit in self.z_ancillas.items():
                x, y = transform_x(coord[0]), transform_y(coord[1])
                svg_parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="#4dabf7"/>')
                if numbering:
                    svg_parts.append(f'<text x="{x}" y="{y + font_size//3}" font-size="{font_size}" font-family="monospace" fill="white" text-anchor="middle">{qubit.id}</text>')
            
            svg_parts.append('</svg>')
            svg_string = '\n'.join(svg_parts)
        
        # Return IPython SVG object if in Jupyter, otherwise return string
        try:
            from IPython.display import SVG
            return SVG(svg_string)
        except ImportError:
            return svg_string

    def draw_checks(self):
        """
        Draws the CNOT gates for each check of the repetition code as SVG diagrams.

        Returns:
            IPython.display.SVG or str: SVG display object showing both checks (2 rounds).
        """
        # Find max ID length for sizing
        all_ids = [q.id for q in self.data_qubits.values()] + \
                  [q.id for q in self.z_ancillas.values()]
        max_id_digits = len(str(max(all_ids))) if all_ids else 1
        
        # Adaptive radius based on number of digits
        if max_id_digits == 1:
            radius = 10
            font_size = 10
        elif max_id_digits == 2:
            radius = 12
            font_size = 9
        else:  # 3+ digits
            radius = 15
            font_size = 8
        
        # Get bounds
        all_coords = list(self.data_qubits.keys()) + list(self.z_ancillas.keys())
        min_x = min(c[0] for c in all_coords)
        max_x = max(c[0] for c in all_coords)
        min_y = min(c[1] for c in all_coords)
        max_y = max(c[1] for c in all_coords)
        
        scale = 30
        padding = 50  # Extra padding for labels
        single_width = int((max_x - min_x) * scale + 2 * padding)
        single_height = int((max_y - min_y) * scale + 2 * padding)
        
        # Total SVG size (1x2 horizontal layout for 2 checks)
        spacing = 10
        total_width = single_width * 2 + spacing * 3
        total_height = single_height + spacing * 2
        
        def transform_x(x, offset_x=0):
            return (x - min_x) * scale + padding + offset_x
        
        def transform_y(y, offset_y=0):
            return (y - min_y) * scale + padding + offset_y
        
        svg_parts = [f'<svg width="{total_width}" height="{total_height}" xmlns="http://www.w3.org/2000/svg">']
        
        # Draw 2 checks side by side
        for check_num in range(2):
            offset_x = check_num * (single_width + spacing) + spacing
            offset_y = spacing
            
            # Title
            svg_parts.append(f'<text x="{offset_x + single_width//2}" y="{offset_y + 20}" font-size="14" font-weight="bold" font-family="monospace" text-anchor="middle">CHECK {check_num + 1}</text>')

            # Draw horizontal line through data qubits
            data_coords = sorted(self.data_qubits.keys())
            if len(data_coords) >= 2:
                x_start = transform_x(data_coords[0][0], offset_x)
                x_end = transform_x(data_coords[-1][0], offset_x)
                y_line = transform_y(data_coords[0][1], offset_y)
                svg_parts.append(f'<line x1="{x_start}" y1="{y_line}" x2="{x_end}" y2="{y_line}" stroke="#e0e0e0" stroke-width="3"/>')
            
            # Collect qubits involved in CNOTs for this check
            cnot_qubits = set()
            for gate in self.gates[check_num]:
                cnot_qubits.add(gate[0])
                cnot_qubits.add(gate[1])
            
            # Draw all qubits not involved in CNOTs
            for coord, qubit in self.data_qubits.items():
                if qubit.id not in cnot_qubits:
                    x, y = transform_x(coord[0], offset_x), transform_y(coord[1], offset_y)
                    svg_parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="black"/>')
                    svg_parts.append(f'<text x="{x}" y="{y + font_size//3}" font-size="{font_size}" font-family="monospace" fill="white" text-anchor="middle">{qubit.id}</text>')
            
            for coord, qubit in self.z_ancillas.items():
                if qubit.id not in cnot_qubits:
                    x, y = transform_x(coord[0], offset_x), transform_y(coord[1], offset_y)
                    svg_parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="#4dabf7"/>')
                    svg_parts.append(f'<text x="{x}" y="{y + font_size//3}" font-size="{font_size}" font-family="monospace" fill="white" text-anchor="middle">{qubit.id}</text>')
            
            # Draw CNOT gates
            for gate in self.gates[check_num]:
                control = self.qubit_with_id(gate[0])
                target = self.qubit_with_id(gate[1])
                
                # Z-ancilla is always the target for repetition code
                color = '#4dabf7'
                
                cx = transform_x(control.coords[0], offset_x)
                cy = transform_y(control.coords[1], offset_y)
                tx = transform_x(target.coords[0], offset_x)
                ty = transform_y(target.coords[1], offset_y)
                
                # Calculate line endpoints to stop at target edge
                target_radius = 8
                dx = tx - cx
                dy = ty - cy
                length = (dx**2 + dy**2)**0.5
                if length > 0:
                    dx_norm = dx / length
                    dy_norm = dy / length
                    line_start_x = cx + dx_norm * 5
                    line_start_y = cy + dy_norm * 5
                    line_end_x = tx - dx_norm * target_radius
                    line_end_y = ty - dy_norm * target_radius
                else:
                    line_start_x, line_start_y = cx, cy
                    line_end_x, line_end_y = tx, ty
                
                # Draw line connecting control and target
                svg_parts.append(f'<line x1="{line_start_x}" y1="{line_start_y}" x2="{line_end_x}" y2="{line_end_y}" stroke="{color}" stroke-width="2"/>')
                
                # Draw control (filled circle)
                svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="{color}"/>')
                
                # Draw target (circle with plus sign)
                svg_parts.append(f'<circle cx="{tx}" cy="{ty}" r="{target_radius}" fill="white" stroke="{color}" stroke-width="2"/>')
                svg_parts.append(f'<line x1="{tx}" y1="{ty - target_radius}" x2="{tx}" y2="{ty + target_radius}" stroke="{color}" stroke-width="2"/>')
                svg_parts.append(f'<line x1="{tx - target_radius}" y1="{ty}" x2="{tx + target_radius}" y2="{ty}" stroke="{color}" stroke-width="2"/>')
                
                # Add qubit ID labels
                label_offset = 12
                svg_parts.append(f'<text x="{cx}" y="{cy - label_offset}" font-size="{font_size}" font-family="monospace" text-anchor="middle">{control.id}</text>')
                svg_parts.append(f'<text x="{tx}" y="{ty + label_offset + font_size}" font-size="{font_size}" font-family="monospace" text-anchor="middle">{target.id}</text>')
        
        svg_parts.append('</svg>')
        svg_string = '\n'.join(svg_parts)
        
        # Return IPython SVG object if in Jupyter, otherwise return string
        try:
            from IPython.display import SVG
            return SVG(svg_string)
        except ImportError:
            return svg_string
    