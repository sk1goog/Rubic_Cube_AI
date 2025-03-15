#!/usr/bin/env python3
import json
import csv
import random
import time
import os
import concurrent.futures
import hashlib  # For hash calculation

# Global script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# Load parameters from CSV file
# ------------------------------------------------------------------
def load_parameters_from_csv(filename="parameter_csv-Parameter.csv"):
    """
    Reads parameters from the CSV file located in the same directory as the script.
    The CSV file contains in its first row the header with the names of parameter sets (e.g., P1, P2, P3).
    For each subsequent row, the parameter name and its corresponding value are read.
    If a composite key (e.g., "MAX_ITERATIONS E1") is found, the first part is used as the main key and the second as a subkey.
    A random parameter set is selected.
    Thousand separators are removed from numeric values and they are converted to int (or float).
    Returns a tuple (parameters, chosen_set).
    """
    filepath = os.path.join(SCRIPT_DIR, filename)
    
    with open(filepath, newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        rows = list(reader)
    
    if not rows:
        raise ValueError("The CSV file is empty.")
    
    header = rows[0]
    num_sets = len(header) - 1  # first column is the parameter name
    
    chosen_index = random.randint(1, num_sets)
    chosen_set = header[chosen_index].strip()
    
    parameters = {}
    for row in rows[1:]:
        if not row or len(row) < chosen_index + 1:
            continue
        key = row[0].strip()
        value_str = row[chosen_index].strip()
        value_str_clean = value_str.replace(".", "")
        try:
            value = int(value_str_clean)
        except ValueError:
            try:
                value = float(value_str_clean)
            except ValueError:
                value = value_str
        
        if " " in key:
            main_key, sub_key = key.split(" ", 1)
            if main_key not in parameters:
                parameters[main_key] = {}
            parameters[main_key][sub_key] = value
        else:
            parameters[key] = value
    
    return parameters, chosen_set

# ------------------------------------------------------------------
# Target definitions and additional configurations
# ------------------------------------------------------------------
# Arrangement of positions (0-based indices):
# Front:   indices 0..8       (positions 1-9)
# Left:    indices 9..17      (positions 10-18)
# Up:      indices 18..26     (positions 19-27)
# Right:   indices 27..35     (positions 28-36)
# Down:    indices 36..44     (positions 37-45)
# Back:    indices 45..53     (positions 46-54)

pieces_level1 = {
    "E1": ((1, 12, 25), ('w', 'r', 'g')),
    "E2": ((3, 27, 28), ('w', 'g', 'o')),
    "E3": ((9, 34, 39), ('w', 'o', 'b')),
    "E4": ((7, 18, 37), ('w', 'r', 'b')),
    "K1": ((4, 15),    ('w', 'r')),
    "K2": ((2, 26),    ('w', 'g')),
    "K3": ((6, 31),    ('w', 'o')),
    "K4": ((8, 38),    ('w', 'b')),
}
pieces_level2 = {
    "K5": ((17, 40), ('r', 'b')),
    "K6": ((11, 22), ('r', 'g')),
    "K7": ((24, 29), ('g', 'o')),
    "K8": ((35, 42), ('o', 'b')),
}
pieces_level3 = {
    "E5":  ((10, 19, 52), ('r', 'g', 'y')),
    "E6":  ((21, 30, 54), ('g', 'o', 'y')),
    "E7":  ((36, 45, 48), ('o', 'b', 'y')),
    "E8":  ((16, 43, 46), ('r', 'b', 'y')),
    "K9":  ((13, 49),     ('r', 'y')),
    "K10": ((20, 53),     ('g', 'y')),
    "K11": ((33, 51),     ('o', 'y')),
    "K12": ((44, 47),     ('b', 'y')),
}
levels = {
    "E1": {**pieces_level1},
    "E2": {**pieces_level1, **pieces_level2},
    "E3": {**pieces_level1, **pieces_level2, **pieces_level3},
}

allowed_moves_global = ["F", "Fb", "U", "Ub", "L", "Lb", "R", "Rb", "B", "Bb", "D", "Db"]

# The total number of runs is set here once (e.g., from a parameter set that is not randomly selected).
# Alternatively, this value could also be hardcoded.
_global_params, _ = load_parameters_from_csv()
TOTAL_RUNS = _global_params["TOTAL_RUNS"]

num_workers = os.cpu_count() or 4
batch_size = num_workers
attempts_per_task = 3000

# Global set to store already learned (canonical) hashes
learned_hashes = set()
improvements_filename = os.path.join(SCRIPT_DIR, "Improvements.csv")
if os.path.exists(improvements_filename):
    with open(improvements_filename, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            if "Hash" in row:
                learned_hashes.add(row["Hash"])

# ------------------------------------------------------------------
# Functions for rotation transposition
# ------------------------------------------------------------------
def rotate_face(face):
    rotated_face = [face[6], face[3], face[0],
                    face[7], face[4], face[1],
                    face[8], face[5], face[2]]
    return rotated_face

def rotate_state(cube_state, k):
    front = cube_state[0:9]
    left  = cube_state[9:18]
    up    = cube_state[18:27]
    right = cube_state[27:36]
    down  = cube_state[36:45]
    back  = cube_state[45:54]
    
    for _ in range(k % 4):
        new_front = rotate_face(left)
        new_right = rotate_face(front)
        new_back  = rotate_face(right)
        new_left  = rotate_face(back)
        up = rotate_face(up)
        down = rotate_face(down)
        front, left, right, back = new_front, new_left, new_right, new_back
    return front + left + up + right + down + back

def canonical_state_hash(cube_state):
    states = []
    for k in range(4):
        rotated = rotate_state(cube_state, k)
        state_str = ''.join(rotated)
        states.append(state_str)
    canonical = min(states)
    return hashlib.md5(canonical.encode('utf-8')).hexdigest()

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def load_cube_from_csv():
    csv_path = os.path.join(SCRIPT_DIR, "csv_export-StartPos.csv")
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError("CSV file must contain at least two rows.")
    colors = rows[1]
    if len(colors) != 54:
        raise ValueError(f"Expected 54 color codes, found: {len(colors)}")
    return colors

def load_mappings(filename="mappings.json"):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_inverse_dict(moves_mapping):
    return {
        move: (move[:-1] if move.endswith("b") else move + "b")
        for move in moves_mapping
        if (move[:-1] if move.endswith("b") else move + "b") in moves_mapping
    }

def apply_move(cube_state, move, moves_mapping):
    new_state = cube_state.copy()
    for src_str, tgt in moves_mapping.get(move, {}).items():
        new_state[int(tgt) - 1] = cube_state[int(src_str) - 1]
    return new_state

def apply_sequence(cube_state, sequence, moves_mapping):
    for move in sequence:
        cube_state = apply_move(cube_state, move, moves_mapping)
    return cube_state

def is_inverse(move1, move2, inverse_dict):
    return inverse_dict.get(move1) == move2

def cube_to_string(cube_state):
    def col(pos):
        return cube_state[pos - 1]
    lines = [
        f"        {col(19)} {col(20)} {col(21)}",
        f"        {col(22)} {col(23)} {col(24)}",
        f"        {col(25)} {col(26)} {col(27)}",
        f"{col(10)} {col(11)} {col(12)} | {col(1)} {col(2)} {col(3)} | {col(28)} {col(29)} {col(30)}",
        f"{col(13)} {col(14)} {col(15)} | {col(4)} {col(5)} {col(6)} | {col(31)} {col(32)} {col(33)}",
        f"{col(16)} {col(17)} {col(18)} | {col(7)} {col(8)} {col(9)} | {col(34)} {col(35)} {col(36)}",
        f"        {col(37)} {col(38)} {col(39)}",
        f"        {col(40)} {col(41)} {col(42)}",
        f"        {col(43)} {col(44)} {col(45)}",
        f"        {col(46)} {col(47)} {col(48)}",
        f"        {col(49)} {col(50)} {col(51)}",
        f"        {col(52)} {col(53)} {col(54)}"
    ]
    return "\n".join(lines)

log_lines = []
def log_and_print(message):
    print(message)
    log_lines.append(message)

def get_correct_pieces(cube_state, target_pieces):
    return {
        piece for piece, (positions, colors) in target_pieces.items()
        if all(cube_state[pos - 1] == col for pos, col in zip(positions, colors))
    }

def count_correct_pieces(cube_state, target_pieces):
    return len(get_correct_pieces(cube_state, target_pieces))

def format_with_thousands(number):
    return f"{number:,}".replace(",", ".")

def calculate_move_damage(state, locked, move, moves_mapping, target_pieces):
    mapping = moves_mapping.get(move, {})
    affected = 0
    for piece in locked:
        positions, _ = target_pieces[piece]
        if any(str(pos) in mapping for pos in positions):
            affected += 1
    return affected

def compute_allowed_moves_mix(allowed_moves_global, variability):
    pairs = []
    used = set()
    for move in allowed_moves_global:
        if move in used:
            continue
        inverse = move[:-1] if move.endswith("b") else move + "b"
        if inverse in allowed_moves_global:
            pairs.append((move, inverse))
            used.add(move)
            used.add(inverse)
        else:
            pairs.append((move,))
            used.add(move)
    total_pairs = len(pairs)
    num_pairs = max(1, round((variability / 100.0) * total_pairs))
    selected_pairs = random.sample(pairs, num_pairs)
    allowed_moves_mix = [move for pair in selected_pairs for move in pair]
    return allowed_moves_mix

def generate_random_sequence_custom(length, allowed_moves_mix, inverse_dict, damaging_move):
    sequence = []
    first_candidates = [m for m in allowed_moves_mix if m != damaging_move]
    if not first_candidates:
        first_candidates = allowed_moves_mix
    first_move = random.choice(first_candidates)
    sequence.append(first_move)
    while len(sequence) < length:
        move = random.choice(allowed_moves_mix)
        if sequence and is_inverse(sequence[-1], move, inverse_dict):
            continue
        sequence.append(move)
    return sequence

def candidate_search_bundle(best_state, current_move_length, inverse_dict, moves_mapping,
                            level_targets, best_count, locked, attempts, variability):
    best_candidate = None
    for _ in range(attempts):
        allowed_moves_mix = compute_allowed_moves_mix(allowed_moves_global, variability)
        damaging_move = None
        max_damage = -1
        for move in allowed_moves_mix:
            damage = calculate_move_damage(best_state, locked, move, moves_mapping, level_targets)
            if damage > max_damage:
                max_damage = damage
                damaging_move = move
        seq = generate_random_sequence_custom(current_move_length, allowed_moves_mix, inverse_dict, damaging_move)
        candidate_state = apply_sequence(best_state, seq, moves_mapping)
        if not locked.issubset(get_correct_pieces(candidate_state, level_targets)):
            continue
        candidate_count = count_correct_pieces(candidate_state, level_targets)
        if candidate_count > best_count:
            best_candidate = (candidate_count, candidate_state, seq)
    return best_candidate

# ------------------------------------------------------------------
# Logging improvements including canonical state hash
# ------------------------------------------------------------------
def log_improvement(
    csv_file_path,
    date_str,
    time_str,
    correct_before,
    correct_after,
    seq,
    pieces_before_str,
    pieces_after_str,
    improved_count,
    ratio,
    state_hash
):
    if state_hash in learned_hashes:
        return False
    learned_hashes.add(state_hash)
    
    file_exists = os.path.exists(csv_file_path)
    with open(csv_file_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        if not file_exists:
            writer.writerow([
                "Date",
                "Time",
                "Correct Pieces Before Improvement",
                "Correct Pieces After Improvement",
                "Move Sequence",
                "Number of Moves",
                "Correct Positions Before",
                "Correct Positions After",
                "Number of Improved Positions",
                "Value",
                "Hash"
            ])
        writer.writerow([
            date_str,
            time_str,
            correct_before,
            correct_after,
            " ".join(seq),
            len(seq),
            pieces_before_str,
            pieces_after_str,
            improved_count,
            f"{ratio:.2f}",
            state_hash
        ])
    return True

# ------------------------------------------------------------------
# Main program with multi-processing and run loop
# ------------------------------------------------------------------
def main():
    results_filename = os.path.join(SCRIPT_DIR, "Results.csv")
    if not os.path.exists(results_filename):
        with open(results_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([
                "Date", "Start Run Timestamp", "Parameter Set", "Run Number",
                "Run Duration (hh:mm:ss)", "Total Correct Pieces",
                "Last Improvement Iteration", "Sequence Length", "Total Move Sequence", "New Moves"
            ])
    
    # Create the ProcessPoolExecutor once:
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for run in range(1, TOTAL_RUNS + 1):
            # For each run, a random parameter set is selected:
            params, chosen_set = load_parameters_from_csv()
            # Set the relevant parameters for this run:
            MAX_ITERATIONS = params["MAX_ITERATIONS"]
            MAX_MOVES_PER_SEQUENCE = params["MAX_MOVES_PER_SEQUENCE"]
            TARGET_CORRECT_STONES = params["TARGET_CORRECT_STONES"]
            NO_IMPROVEMENT_THRESHOLD = params["NO_IMPROVEMENT_THRESHOLD"]
            VARIABLITY = params["VARIABLITY"]
            
            new_learned_count = 0
            global log_lines
            log_lines = []
            start_time = time.time()
            run_date = time.strftime("%Y-%m-%d", time.localtime(start_time))
            run_start_time = time.strftime("%H:%M:%S", time.localtime(start_time))
            last_improvement_iteration = 0
            
            log_and_print(f"===== Run {run} of {TOTAL_RUNS} =====")
            log_and_print(f"Using parameter set: {chosen_set}")
            
            try:
                current_state = load_cube_from_csv()
            except Exception as e:
                log_and_print(f"Error reading starting cube position from CSV file: {e}")
                continue
            
            log_and_print("----- Starting cube position -----")
            log_and_print(cube_to_string(current_state))
            start_count = count_correct_pieces(current_state, levels["E1"])
            log_and_print(f"Number of correctly positioned pieces (E1): {start_count}")
            
            try:
                moves_mapping = load_mappings("mappings.json")
            except Exception as e:
                log_and_print(f"Error loading mappings.json: {e}")
                continue
            
            inverse_dict = generate_inverse_dict(moves_mapping)
            total_move_sequence = []
            run_best_count = count_correct_pieces(current_state, levels["E1"])
            run_best_sequence = []
            log_and_print(f"Starting solution search ...\nAvailable CPU cores: {os.cpu_count() or 4}")
            
            for level in ["E1", "E2", "E3"]:
                log_and_print(f"\n----- Solution for level {level} -----")
                max_iter = MAX_ITERATIONS[level]
                target = TARGET_CORRECT_STONES[level]
                max_moves_allowed = MAX_MOVES_PER_SEQUENCE[level]
                no_improvement_threshold = NO_IMPROVEMENT_THRESHOLD[level]
                level_targets = levels[level]
            
                best_count = count_correct_pieces(current_state, level_targets)
                best_state = current_state.copy()
                best_sequence = []
                locked = get_correct_pieces(best_state, level_targets)
            
                current_move_length = 1
                no_improvement_counter = 0
                iteration = 0
            
                while best_count < target and iteration < max_iter:
                    iteration += batch_size * attempts_per_task
                    if iteration % 1_000_000 == 0:
                        log_and_print(f"Iterations so far: {format_with_thousands(iteration)}")
                    
                    results = []
                    futures = [
                        executor.submit(
                            candidate_search_bundle,
                            best_state,
                            current_move_length,
                            inverse_dict,
                            moves_mapping,
                            level_targets,
                            best_count,
                            locked,
                            attempts_per_task,
                            VARIABLITY
                        )
                        for _ in range(batch_size)
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        res = future.result()
                        if res is not None:
                            results.append(res)
                    
                    if results:
                        best_candidate = max(results, key=lambda r: r[0])
                        candidate_count, candidate_state, candidate_seq = best_candidate
                        
                        if candidate_count > best_count:
                            before_correct_pieces = get_correct_pieces(best_state, level_targets)
                            before_correct_pieces_str = "-".join(sorted(list(before_correct_pieces)))
                            correct_before = len(before_correct_pieces)
                            
                            after_correct_pieces = get_correct_pieces(candidate_state, level_targets)
                            after_correct_pieces_str = "-".join(sorted(list(after_correct_pieces)))
                            correct_after = len(after_correct_pieces)
                            
                            improvement_count = correct_after - correct_before
                            ratio = improvement_count / len(candidate_seq) if len(candidate_seq) > 0 else 0
                            
                            state_hash = canonical_state_hash(best_state)
                            
                            improvement_date = time.strftime("%Y-%m-%d", time.localtime())
                            improvement_time = time.strftime("%H:%M:%S", time.localtime())
                            
                            if log_improvement(
                                improvements_filename,
                                improvement_date,
                                improvement_time,
                                correct_before,
                                correct_after,
                                candidate_seq,
                                before_correct_pieces_str,
                                after_correct_pieces_str,
                                improvement_count,
                                ratio,
                                state_hash
                            ):
                                new_learned_count += 1
                            
                            best_count = candidate_count
                            best_sequence.extend(candidate_seq)
                            best_state = candidate_state
                            locked = after_correct_pieces
                            current_move_length = 1
                            no_improvement_counter = 0
                            last_improvement_iteration = iteration
                            
                            log_and_print(
                                f"\nNew best position (Level {level}) at iteration {format_with_thousands(iteration)}:\n"
                                f"Cube state:\n{cube_to_string(best_state)}\n"
                                f"Number of correctly positioned pieces: {best_count}\n"
                                f"Total move sequence: {' '.join(total_move_sequence + best_sequence)}\n"
                                f"Number of moves: {len(total_move_sequence + best_sequence)}"
                            )
                        
                        if best_count == target:
                            break
                    else:
                        no_improvement_counter += batch_size * attempts_per_task
                    
                    if no_improvement_counter >= no_improvement_threshold and current_move_length < max_moves_allowed:
                        current_move_length += 1
                        no_improvement_counter = 0
                
                if best_count < target:
                    log_and_print(
                        f"\nLevel {level} could not be completely solved after {format_with_thousands(iteration)} attempts.\n"
                        f"Last cube state:\n{cube_to_string(best_state)}\n"
                        f"Number of correctly positioned pieces: {best_count}"
                    )
                    run_best_count = best_count
                    run_best_sequence = total_move_sequence + best_sequence
                    break
                else:
                    current_state = best_state.copy()
                    total_move_sequence += best_sequence
                    run_best_count = best_count
                    run_best_sequence = total_move_sequence
            
            elapsed_time = time.time() - start_time
            solution_str = " ".join(run_best_sequence) if run_best_sequence else "No move sequence"
            final_correct = run_best_count
            
            log_and_print(
                f"\n*** Final result for Run {run} ***\n"
                f"Cube state:\n{cube_to_string(current_state)}\n\n"
                f"Number of correctly positioned pieces (E3 i.e., best solution): {final_correct}\n"
                f"Total move sequence: {solution_str}\n"
                f"Number of moves: {len(run_best_sequence)}\n"
                f"Runtime: {elapsed_time:.2f} seconds"
            )
            
            runtime_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            output_filename = f"{run_date}_{os.path.splitext(os.path.basename(__file__))[0]}_Run{run}_CP-{final_correct}_RT-{runtime_str}.txt"
            
            with open(output_filename, "w", encoding="utf-8") as f:
                for line in log_lines:
                    f.write(line + "\n")
            
            log_and_print(f"\nThe results for Run {run} have been saved in '{output_filename}'.\n")
            
            last_improvement_str = format_with_thousands(last_improvement_iteration)
            sequence_length = format_with_thousands(len(run_best_sequence))
            
            with open(results_filename, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow([
                    run_date,
                    run_start_time,
                    chosen_set,
                    run,
                    runtime_str,
                    final_correct,
                    last_improvement_str,
                    sequence_length,
                    solution_str,
                    new_learned_count
                ])
        
    print("All runs completed.")

if __name__ == '__main__':
    main()