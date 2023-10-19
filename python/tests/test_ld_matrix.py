from dataclasses import dataclass, field
from enum import Enum
from itertools import permutations, combinations_with_replacement
from typing import List
import io

import numpy as np
import pytest

import tskit


class NormMethod(Enum):
    """
    Method for norming the output statistic

    TOTAL: Divide the statistic by the total number of haplotypes
    HAP_WEIGHTED: Weight each allele's statistic by the proportion of the haplotype present
    AF_WEIGHTED: Weight each allele's statistic by the product of the allele frequencies
    """

    TOTAL = "total"
    HAP_WEIGHTED = "hap_weighted"
    AF_WEIGHTED = "af_weighted"


def get_state(ts, idx):
    state = np.zeros((len(idx.sites), ts.num_samples), dtype=int)
    current_site = 0
    for tree in ts.trees():
        for site in tree.sites():
            if site.id in idx.sites:
                states = [site.ancestral_state]
                current_state = 0
                for mutation in site.mutations:
                    # if we've seen the state, use the index as the enumerated
                    # state value, otherwise, add another state value
                    if mutation.derived_state in states:
                        current_state = states.index(mutation.derived_state)
                    else:
                        states.append(mutation.derived_state)
                        current_state = len(states) - 1

                    for sample_id in tree.samples(mutation.node):
                        state[current_site, sample_id] = current_state
                current_site += 1
    return state


def compute_stat_and_weights(
    hap_mat,
    sample_sets,
    summary_func,
    polarized,
    norm_method,
    print_weights,
):
    hap_mat = np.asarray(hap_mat)

    # number of samples
    n = hap_mat.sum()
    # number of B alleles, number of A alleles
    n_a, n_b = hap_mat.shape

    weights = np.zeros(hap_mat.shape)
    stats = np.zeros(hap_mat.shape)
    for a_idx in range(1 if polarized else 0, n_a):
        for b_idx in range(1 if polarized else 0, n_b):
            w_AB = hap_mat[a_idx, b_idx]
            w_Ab = hap_mat[a_idx, :].sum() - w_AB
            w_aB = hap_mat[:, b_idx].sum() - w_AB

            stats[a_idx, b_idx] = summary_func(w_AB, w_Ab, w_aB, n)

            if print_weights:
                print(a_idx, b_idx, int(w_AB), int(w_Ab), int(w_aB), int(n), sep="\t")
            # create weights matrix
            if norm_method is NormMethod.HAP_WEIGHTED:
                hap_freq = hap_mat / n
                weights[a_idx, b_idx] = hap_freq[a_idx, b_idx]
            elif norm_method is NormMethod.AF_WEIGHTED:
                p_a = hap_mat.sum(1) / n
                p_b = hap_mat.sum(0) / n
                weights[a_idx, b_idx] = p_a[a_idx] * p_b[b_idx]
            elif norm_method is NormMethod.TOTAL:
                weights[a_idx, b_idx] = 1 / (
                    (n_a - (1 if polarized else 0)) * (n_b - (1 if polarized else 0))
                )

    return stats, weights


@dataclass
class MatrixIndex:
    n_rows: int
    n_cols: int
    sites: List[int] = field(default_factory=list)
    rshr: List[int] = field(default_factory=list)
    cshr: List[int] = field(default_factory=list)
    rdiff: List[int] = field(default_factory=list)
    cdiff: List[int] = field(default_factory=list)
    shr_idx: List[int] = field(default_factory=list)
    rdiff_idx: List[int] = field(default_factory=list)
    cdiff_idx: List[int] = field(default_factory=list)


def check_sites(sites, max_sites):
    if sites is None or len(sites) == 0:
        raise ValueError("No sites provided")
    i = 0
    for i in range(len(sites) - 1):
        if sites[i] < 0 or sites[i] >= max_sites:
            raise ValueError(f"Site out of bounds: {sites[i]}")
        if sites[i] >= sites[i + 1]:
            raise ValueError(f"Sites not sorted: {sites[i], sites[i + 1]}")
    if sites[-1] < 0 or sites[-1] >= max_sites:
        raise ValueError(f"Site out of bounds: {sites[i + 1]}")


def get_matrix_indices(row_sites, col_sites):
    r = 0
    c = 0
    s = 0
    idx = MatrixIndex(len(row_sites), len(col_sites))

    while r < len(row_sites) and c < len(col_sites):
        if row_sites[r] < col_sites[c]:
            idx.rdiff_idx.append(s)
            idx.rdiff.append(r)
            idx.sites.append(row_sites[r])
            r += 1
            s += 1
        elif col_sites[c] < row_sites[r]:
            idx.cdiff_idx.append(s)
            idx.cdiff.append(c)
            idx.sites.append(col_sites[c])
            c += 1
            s += 1
        else:
            idx.rshr.append(r)
            idx.cshr.append(c)
            idx.shr_idx.append(s)
            idx.sites.append(row_sites[r])
            r += 1
            c += 1
            s += 1
    if r < len(row_sites):
        while r < len(row_sites):
            idx.rdiff.append(r)
            idx.rdiff_idx.append(s)
            idx.sites.append(row_sites[r])
            r += 1
    if c < len(col_sites):
        while c < len(col_sites):
            idx.cdiff.append(c)
            idx.cdiff_idx.append(s)
            idx.sites.append(col_sites[c])
            c += 1

    return idx


def compute_two_site_general_stat(
    state,
    func,
    polarized,
    norm_method,
    sample_sets,
    idx,
    debug=False,
    print_weights=False,
):
    state = np.asarray(state)
    norm = NormMethod(norm_method)

    result = np.zeros((len(sample_sets), idx.n_rows, idx.n_cols))

    def compute(left_states, right_states):
        hap_mat = np.zeros((np.max(left_states) + 1, np.max(right_states) + 1))
        for A_i, B_i in zip(left_states, right_states):
            hap_mat[A_i, B_i] += 1
        stats, weights = compute_stat_and_weights(
            hap_mat, sample_sets, func, polarized, norm, print_weights
        )
        if debug:
            print("hap_mat", hap_mat, "stats", stats, "weights", weights, "============", sep="\n")
        return (stats * weights).sum()

    for i, ss in enumerate(sample_sets):
        for r in range(len(idx.rdiff)):
            for c in range(len(idx.cdiff)):
                result[i, idx.rdiff[r], idx.cdiff[c]] = compute(
                    state[idx.rdiff_idx[r], ss], state[idx.cdiff_idx[c], ss]
                )
            for c in range(len(idx.shr_idx)):
                result[i, idx.rdiff[r], idx.cshr[c]] = compute(
                    state[idx.rdiff_idx[r], ss], state[idx.shr_idx[c], ss]
                )
        for r in range(len(idx.shr_idx)):
            for c in range(len(idx.cdiff)):
                result[i, idx.rshr[r], idx.cdiff[c]] = compute(
                    state[idx.shr_idx[r], ss], state[idx.cdiff_idx[c], ss]
                )
        inner = 0
        for r in range(len(idx.shr_idx)):
            for c in range(inner, len(idx.shr_idx)):
                result[i, idx.rshr[r], idx.cshr[c]] = compute(
                    state[idx.shr_idx[r], ss], state[idx.shr_idx[c], ss]
                )
                if (idx.sites[idx.shr_idx[r]] != idx.sites[idx.shr_idx[c]]):
                    result[i, idx.rshr[c], idx.cshr[r]] = result[i, idx.rshr[r], idx.cshr[c]] 

    return result


def two_site_general_stat(
    ts,
    summary_func,
    norm_method,
    polarized,
    sites=None,
    sample_sets=None,
    debug=False,
    print_weights=False,
):
    if sample_sets is None:
        sample_sets = [ts.samples()]
    if sites is None:
        sites = [np.arange(ts.num_sites), np.arange(ts.num_sites)]
    else:
        if len(sites) != 2:
            raise ValueError(f"Sites must be a length 2 list, got a length {len(sites)} list")
        sites[0] = np.asarray(sites[0])
        sites[1] = np.asarray(sites[1])

    
    row_sites, col_sites = sites
    check_sites(row_sites, ts.num_sites)
    check_sites(col_sites, ts.num_sites)
    idx = get_matrix_indices(row_sites, col_sites)

    state = get_state(ts, idx)

    result = compute_two_site_general_stat(
        state, summary_func, polarized, norm_method, sample_sets, idx, debug, print_weights
    )
    if len(sample_sets) == 1:
        return result.reshape(result.shape[1:3])
    return result


def r2(w_AB, w_Ab, w_aB, n):
    p_AB = w_AB / float(n)
    p_Ab = w_Ab / float(n)
    p_aB = w_aB / float(n)

    p_A = p_AB + p_Ab
    p_B = p_AB + p_aB

    D_ = p_AB - (p_A * p_B)
    denom = p_A * p_B * (1 - p_A) * (1 - p_B)

    if denom == 0 and D_ == 0:
        return 0.0

    return (D_ * D_) / denom



def get_paper_ex_ts():
    # Data taken from the tests: https://github.com/tskit-dev/tskit/blob/61a844a/c/tests/testlib.c#L55-L96

    nodes = """\
    is_sample time population individual
    1  0       -1   0
    1  0       -1   0
    1  0       -1   1
    1  0       -1   1
    0  0.071   -1   -1
    0  0.090   -1   -1
    0  0.170   -1   -1
    0  0.202   -1   -1
    0  0.253   -1   -1
    """

    edges = """\
    left   right   parent  child
    2 10 4 2
    2 10 4 3
    0 10 5 1
    0 2  5 3
    2 10 5 4
    0 7  6 0,5
    7 10 7 0,5
    0 2  8 2,6
    """

    sites = """\
    position ancestral_state
    1      0
    4.5    0
    8.5    0
    """

    mutations = """\
    site node derived_state
    0      2   1
    1      0   1
    2      5   1
    """

    individuals = """\
    flags  location   parents
    0      0.2,1.5    -1,-1
    0      0.0,0.0    -1,-1
    """

    return tskit.load_text(
        nodes=io.StringIO(nodes),
        edges=io.StringIO(edges),
        sites=io.StringIO(sites),
        individuals=io.StringIO(individuals),
        mutations=io.StringIO(mutations),
        strict=False,
    )


TS_LD_MATRIX = np.array(
    [[1.0,        0.11111111, 0.11111111],
     [0.11111111, 1.0,        1.0],
     [0.11111111, 1.0,        1.0]]
)


def get_all_site_partitions(n):
    """
    TODO: only works for square matricies
    """
    parts = []
    for l in tskit.combinatorics.rule_asc(3):
        for g in set(permutations(l, len(l))):
            p = []
            i = iter(range(n))
            for item in g:
                p.append([next(i) for _ in range(item)])
            parts.append(p)
    combos = []
    for a, b in combinations_with_replacement({tuple(j) for i in parts for j in i}, 2):
        combos.append((a, b))
        combos.append((b, a))
    combos = [[list(a), list(b)] for a, b in set(combos)]
    return combos


def assert_slice_allclose(a, b):
    ts = get_paper_ex_ts()
    np.testing.assert_allclose(
        two_site_general_stat(ts, r2, "hap_weighted", polarized=False, sites=[a, b]),
        TS_LD_MATRIX[a[0]: a[-1] + 1, b[0]: b[-1] + 1]
    )


@pytest.mark.parametrize("partition", get_all_site_partitions(len(TS_LD_MATRIX)))
def test_all_subsets(partition):
    a, b = partition
    print(a, b)
    assert_slice_allclose(a, b)

# two_site_general_stat(get_paper_ex_ts(), r2, 'hap_weighted', polarized=False, sites=[[1], [1, 2]])
