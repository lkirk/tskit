import contextlib
from itertools import combinations
from itertools import combinations_with_replacement
from itertools import product

import demes
import msprime
import numpy as np
import pytest

from tskit import Interval


@contextlib.contextmanager
def suppress_overflow_div0_warning():
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        yield


def expand_dims(arr):
    """
    Expand the dimensions of the provided array (arrays). This helps to control
    the output dimensions of the ld matrix (ie if there's 2 dimensions to
    indexes or sample_sets, we'll get a 3D ld matrix back. This will not be
    necessary in the C implementation because dimension dropping happens in the
    python layer.
    """
    try:
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return np.expand_dims(arr, axis=0)
    except:
        pass
    try:
        arr = [np.asarray(a) for a in arr]
    except Exception as e:
        raise ValueError("Must be a list of 1D array-like") from e
    for a in arr:
        if a.ndim != 1:
            raise ValueError("Must be a list of 1D arrays")
    return arr


def check_bins(bins, seq_len):
    try:
        bins = np.asarray(bins)
    except Exception as e:
        raise ValueError("Bins must be coercible to a 1D array") from e
    if bins.ndim != 1:
        raise ValueError("Bins must be a 1D array")
    if not np.all(bins[:-1] <= bins[1:]):
        raise ValueError("Bins must be sorted")
    if bins[-1] > seq_len:
        raise ValueError(f"Last bin is out of bounds, must be <= L: {bins[-1]}")
    if len(bins) < 2:
        raise ValueError(f"Must have at least 2 bins, got {len(bins)}")
    if (bins < 0).any():
        raise ValueError("Bins must be greater than 0")
    return bins


def construct_ld_matrix(ts, stat, sample_sets, indexes):
    """
    Produce an ld matrix with the same error characteristics as the C version
    Create an LD matrix by starting at the diagonal of each row. This ensures
    that we accumulate error in the same way as we would in the C version. If
    we produce an LD matrix starting from tree 0 at each row, we accumulate a
    different (likely more) amount of error.
    """
    bp = ts.breakpoints(as_array=True)[:-1]
    k = len(sample_sets) if indexes is None else len(indexes)
    out = np.zeros((k, ts.num_trees, ts.num_trees))
    for i, b in enumerate(bp):
        out[0:k, i, i:] = ts.ld_matrix(
            sample_sets=sample_sets,
            indexes=indexes,
            mode="branch",
            stat=stat,
            positions=[[b], bp[i:]],
        )[:, 0, :]  # result is for one row
    return out


def integrate_stat_over_bin(bin, i1, i2, stat):
    bl, br = bin
    # Integration support
    l_support = i2.left - i1.right
    r_support = i2.right - i1.left
    # length of the middle region
    r2_len = min(i1.right - i1.left, i2.right - i2.left)
    # bounds of the middle region
    r2_l_bound = min(i2.left - i1.left, i2.right - i1.right)
    r2_r_bound = r_support - r2_len

    r1_l = min(max(bl, l_support), r2_l_bound)
    r1_r = max(min(br, r2_l_bound), l_support)
    r2_l = min(max(bl, r2_l_bound), r2_r_bound)
    r2_r = max(min(br, r2_r_bound), r2_l_bound)
    r3_l = min(max(bl, r2_r_bound), r_support)
    r3_r = max(min(br, r_support), r2_r_bound)
    return (
        stat
        / (i1.span * i2.span)
        * (
            -1 / 2 * (r1_l - r1_r) * (2 * i1.right - 2 * i2.left + r1_l + r1_r)
            + (r2_r - r2_l) * r2_len
            + 1 / 2 * (r3_l - r3_r) * (2 * i1.left - 2 * i2.right + r3_l + r3_r)
        )
    )


def isect(l1, r1, l2, r2):
    """left open, right closed"""
    return max(l1, l2) < min(r1, r2) or l1 == r2 or l2 == r1


def get_tree_pair_bounds(ivl_l, ivl_r, bins):
    return Interval(
        max(0, ivl_r.left - ivl_l.right),
        min(bins[-1], ivl_r.right - ivl_l.left),
    )


def ld_decay_branch(ts, bins, stat, sample_sets, indexes):
    ld = construct_ld_matrix(ts, stat, sample_sets, indexes)
    dims = (len(indexes or sample_sets), len(bins) - 1)
    result = np.zeros(dims, dtype=float)
    bincount = np.zeros(dims, dtype=int)
    bp = ts.breakpoints(as_array=True)
    bin_ivls = np.fromiter(zip(bins[:-1], bins[1:]), np.dtype((float, 2)))
    for i, j in combinations_with_replacement(range(ts.num_trees), 2):  # upper tri+diag
        ivl_l = Interval(bp[i], bp[i + 1])
        ivl_r = Interval(bp[j], bp[j + 1])
        bounds = get_tree_pair_bounds(ivl_l, ivl_r, bins)
        for k in range(dims[0]):
            result[k] += np.apply_along_axis(
                integrate_stat_over_bin, 1, bin_ivls, ivl_l, ivl_r, ld[k, i, j]
            )
            bincount[k] += np.fromiter((isect(*bounds, *b) for b in bin_ivls), int)
    if dims[0] == 1:  # drop dims if first dim is length 1
        return result.reshape(dims[1:]), bincount.reshape(dims[1:])
    return result, bincount


def ld_decay_site(ts, bins, stat, sample_sets, indexes):
    # __import__("ipdb").set_trace()
    ld = ts.ld_matrix(stat=stat, sample_sets=sample_sets, indexes=indexes)
    dims = (len(indexes or sample_sets), len(bins) - 1)
    result = np.zeros(dims, dtype=float)
    bincount = np.zeros(dims, dtype=int)
    site_pos = ts.sites_position
    for i in range(ts.num_sites):
        for j in range(i + 1, ts.num_sites):  # upper tri (-diag)
            dist = site_pos[j] - site_pos[i]
            if dist > bins[-1]:
                break
            bin = np.searchsorted(bins[1:], dist, side="left")
            for k in range(dims[0]):
                s = ld[k, i, j]
                if np.isnan(s):
                    continue
                result[k, bin] += s
                bincount[k, bin] += 1
    if dims[0] == 1:  # drop dims if first dim is length 1
        return result.reshape(dims[1:]), bincount.reshape(dims[1:])
    return result, bincount


def ld_decay(
    ts,
    bins,
    stat="r2",
    sample_sets=None,
    indexes=None,
    mode="site",
    return_counts=False,
):
    bins = check_bins(bins, ts.sequence_length)
    sample_sets = expand_dims(sample_sets or [ts.samples()])
    if indexes is not None:
        indexes = expand_dims(indexes)
    match mode:
        case "site":
            result, count = ld_decay_site(ts, bins, stat, sample_sets, indexes)
        case "branch":
            result, count = ld_decay_branch(ts, bins, stat, sample_sets, indexes)
        case _:
            raise ValueError(f"Unknown Stats Mode: {mode}")

    if return_counts:
        return result, count
    with suppress_overflow_div0_warning():
        return result / count


ONE_WAY_STATS = [
    "r",
    "r2",
    "D",
    "D2",
    "D_prime",
    "pi2",
    "Dz",
    "D2_unbiased",
    "Dz_unbiased",
    "pi2_unbiased",
]

TWO_WAY_STATS = ["r2", "D2", "D2_unbiased"]

TS = msprime.sim_mutations(
    msprime.sim_ancestry(
        samples=100,
        sequence_length=1e5,
        recombination_rate=1e-8,
        demography=msprime.Demography.from_demes(
            demes.loads("""
            time_units: generations
            demes:
              - name: A
                epochs:
                  - {start_size: 5000, end_time: 1000}
                  - {start_size: 1000, end_time: 400}
                  - {start_size: 5000, end_time: 0}
            """)
        ),
        random_seed=23,
    ),
    rate=1e-7,
    random_seed=23,
)


@pytest.mark.parametrize("stat,mode", product(ONE_WAY_STATS, ["site", "branch"]))
def test_ld_decay(stat, mode):
    bins = np.logspace(0, np.log10(TS.sequence_length), num=35)
    bins[0] = 0
    decay, counts = ld_decay(TS, bins, stat=stat, mode=mode, return_counts=True)
    c = TS.ld_decay(bins, stat=stat, mode=mode)
    with suppress_overflow_div0_warning():
        np.testing.assert_array_equal(decay / counts, c)
    # Verify that the sum of all LD in our bins is equal to the sum of the LD
    # matrix entries from which they originated.
    if mode == "branch":
        tu = np.triu(
            construct_ld_matrix(
                TS, sample_sets=expand_dims(TS.samples()), indexes=None, stat=stat
            ).squeeze()
        )
        dmask = np.diag_indices_from(tu)
        tu[dmask] = tu[dmask] / 2  # we take half the density on the diagonal
        np.testing.assert_allclose(np.nansum(decay), np.nansum(tu))
        # all but r2 D2 Dz are within 1 ulp
        np.testing.assert_array_almost_equal_nulp(
            np.nansum(decay), np.nansum(tu), nulp=2
        )
    elif mode == "site":
        tu = TS.ld_matrix(stat=stat)[np.triu_indices(TS.num_sites, k=1)]
        np.testing.assert_allclose(decay.sum(), np.nansum(tu))


@pytest.mark.parametrize("stat,mode", product(ONE_WAY_STATS, ["site", "branch"]))
def test_ld_decay_sample_sets(stat, mode):
    bins = np.logspace(0, np.log10(TS.sequence_length), num=35)
    bins[0] = 0
    sample_sets = [TS.samples(), TS.samples(), TS.samples()]
    decay = TS.ld_decay(bins, sample_sets=sample_sets, stat=stat, mode=mode)
    np.testing.assert_array_equal(decay[0], decay[1])
    np.testing.assert_array_equal(decay[1], decay[2])


@pytest.mark.slow
@pytest.mark.parametrize("stat,mode", product(TWO_WAY_STATS, ["site", "branch"]))
def test_two_way_ld_decay(stat, mode):
    bins = np.logspace(0, np.log10(TS.sequence_length), num=35)
    np.testing.assert_array_almost_equal(
        ld_decay(TS, bins, stat=stat, mode=mode),
        TS.ld_decay(bins, stat=stat, mode=mode),
    )
    ss = [TS.samples()] * 3
    indexes = [(0, 0), (0, 1), (1, 1)]
    np.testing.assert_array_almost_equal(
        ld_decay(TS, bins, stat=stat, mode=mode, sample_sets=ss, indexes=indexes),
        TS.ld_decay(bins, stat=stat, mode=mode, sample_sets=ss, indexes=indexes),
    )
