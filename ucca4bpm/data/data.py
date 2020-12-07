import pickle
from scipy import sparse as sp


def matrix_equal(m1: sp.csr_matrix, m2: sp.csr_matrix):
    return (m1 != m2).nnz == 0


def _structure_equal(s1, s2):
    for m1, m2 in zip(s1, s2):
        yield matrix_equal(m1, m2)


def structure_equal(s1, s2):
    if len(s1) != len(s2):
        return False
    return all(_structure_equal(s1, s2))


def is_structure_known(known_structures, structure):
    for idx, (s2, usages) in enumerate(known_structures):
        if structure_equal(structure, s2):
            known_structures[idx] = (s2, usages + 1)
            return True
    return False


DATA_SET_PATH = 'C:\\workspace\\ucca-4-bpm\\ucca4bpm\\data\\transformed\\ucca-output.pickle'
with open(DATA_SET_PATH, 'rb') as f:
    data = pickle.load(f)

structures = data['adjacencies']
known_structures = []

for i, structure in enumerate(structures):
    print(f'Structure {i}/{len(structures)} analyzed.')
    if not is_structure_known(known_structures, structure):
        known_structures.append((structure, 1))

print(f'Got {len(known_structures)} types of matrices, usage below:')
for structure, usage in known_structures:
    print(f'{usage}')
