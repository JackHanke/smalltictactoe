import csv
from itertools import product

def get_combinations(n):
    return [list(p) for p in product([0, 1], repeat=n)]

def compare_terms(term1, term2):
    """Compares two terms and returns the combined term if they differ by one bit."""
    diff = 0
    res = ""
    for i in range(len(term1)):
        if term1[i] != term2[i]:
            diff += 1
            res += "-"
        else:
            res += term1[i]
    return res if diff == 1 else None

def get_prime_implicants(minterms):
    """Step 1 & 2: Grouping and finding Prime Implicants."""
    groups = {}
    for m in minterms:
        ones = m.count('1')
        groups.setdefault(ones, set()).add(m)
    
    prime_implicants = set()
    while groups:
        next_groups = {}
        marked = set()
        keys = sorted(groups.keys())
        
        for i in range(len(keys) - 1):
            for t1 in groups[keys[i]]:
                for t2 in groups[keys[i+1]]:
                    combined = compare_terms(t1, t2)
                    if combined:
                        next_groups.setdefault(keys[i], set()).add(combined)
                        marked.add(t1)
                        marked.add(t2)
        
        for g in groups.values():
            for term in g:
                if term not in marked:
                    prime_implicants.add(term)
        groups = next_groups
        
    return prime_implicants

def solve_pi_table(minterms, prime_implicants):
    """Step 3: Essential Prime Implicant selection (Simplified)."""
    # Map which minterms are covered by which prime implicants
    chart = {m: [] for m in minterms}
    for pi in prime_implicants:
        for m in minterms:
            match = True
            for i in range(len(pi)):
                if pi[i] != '-' and pi[i] != m[i]:
                    match = False
                    break
            if match:
                chart[m].append(pi)
    
    essential = set()
    # Find minterms covered by only one PI
    for m, pis in chart.items():
        if len(pis) == 1:
            essential.add(pis[0])
            
    return essential if essential else prime_implicants

def format_expression(pis, headers):
    """Converts the binary/dash format into a readable Boolean string."""
    final_terms = []
    for pi in pis:
        parts = []
        for i, char in enumerate(pi):
            if char == '1':
                parts.append(headers[i])
            elif char == '0':
                parts.append(f"{headers[i]}'")
        final_terms.append("".join(parts))
    return " + ".join(final_terms)

def run_quine_mccluskey(csv_path):
    with open(csv_path, 'r') as f:
        reader = list(csv.reader(f))
        headers = reader[0][:-1]
        data = reader[1:]

    num_vars = len(headers)
    provided_rows = {}
    for row in data:
        # Convert list of bits to a string key
        key = "".join(row[:-1])
        provided_rows[key] = int(row[-1])

    all_possible = ["".join(map(str, p)) for p in product([0, 1], repeat=num_vars)]
    
    minterms = []
    dont_cares = []

    for bit_str in all_possible:
        if bit_str in provided_rows:
            if provided_rows[bit_str] == 1:
                minterms.append(bit_str)
        else:
            # Missing from CSV = Don't Care
            dont_cares.append(bit_str)

    if not minterms:
        return "Output is always 0"

    # For PI calculation, we treat Don't Cares as 1s
    pi_candidates = get_prime_implicants(minterms + dont_cares)
    # But when solving the chart, we ONLY care about covering the actual minterms
    final_pis = solve_pi_table(minterms, pi_candidates)
    
    return format_expression(final_pis, headers)


if __name__ == '__main__':
    # --- Example Usage ---
    # Assuming 'logic_data.csv' exists
    print(run_quine_mccluskey(f'data/datasets/{8}_moves.csv'))
