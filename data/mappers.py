# data/mappers.py

import torch

class MSCMappers:
    def __init__(self, root_codes, mid_codes, full_codes):
        self.root_to_idx = {c: i for i, c in enumerate(sorted(root_codes))}
        self.mid_to_idx  = {c: i for i, c in enumerate(sorted(mid_codes))}
        self.full_to_idx = {c: i for i, c in enumerate(sorted(full_codes))}

        self.n_root = len(self.root_to_idx)
        self.n_mid  = len(self.mid_to_idx)
        self.n_full = len(self.full_to_idx)


    def root(self, codes):
        return self._multi_hot(codes, self.root_to_idx, self.n_root)

    def mid(self, codes):
        return self._multi_hot(codes, self.mid_to_idx, self.n_mid)

    def full(self, codes):
        return self._multi_hot(codes, self.full_to_idx, self.n_full)
    
    def build_mid_to_full(self):
        mid_to_full = {i: [] for i in range(self.n_mid)}

        for full_code, full_idx in self.full_to_idx.items():
            mid_code = full_code.split('.')[0]
            if mid_code in self.mid_to_idx:
                mid_idx = self.mid_to_idx[mid_code]
                mid_to_full[mid_idx].append(full_idx)
        
        return mid_to_full

    def _multi_hot(self, codes, mapping, size):
        vec = torch.zeros(size, dtype=torch.float32)

        if not codes:
            return vec

        for c in codes:
            if isinstance(c, list):
                for cc in c:
                    if cc in mapping:
                        vec[mapping[cc]] = 1.0
            else:
                if c in mapping:
                    vec[mapping[c]] = 1.0

        return vec



