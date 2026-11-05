"""
Shape of point clouds (e.g. one cell type's embeddings) without a target shape:
persistent homology and intrinsic dimension.

  normalize            center and scale to unit root-mean-square radius
  lifetimes            lifetimes of H1, H2, ... features (Vietoris-Rips, ripser)
  matched_gaussian     Gaussian cloud with the same size and PCA spectrum
  null_lifetimes       lifetimes of the longest features in matched Gaussian clouds
  participation_ratio  PCA participation ratio
  two_nn_dimension     TwoNN intrinsic dimension (Facco et al. 2017)

Signatures: a ring has one long-lived H1 feature; a torus two H1 and one H2.
Needs ripser (pip install ripser).
"""
import numpy as np


def normalize(x):
    """Center and scale to unit root-mean-square distance from the center."""
    x = x - x.mean(0)
    return x / np.sqrt(np.mean(np.sum(x**2, 1)))


def subsample(x, max_points, rng):
    """Random subset of at most max_points rows."""
    if len(x) <= max_points:
        return x
    return x[rng.choice(len(x), max_points, replace=False)]


def lifetimes(x, maxdim=2):
    """
    Persistent homology of a point cloud.

    Returns:
        life: dict dim -> finite lifetimes (death - birth) of the H_dim features,
              longest first, for dim = 1 .. maxdim
        diagrams: the persistence diagrams from ripser (one per dim, 0 .. maxdim)
    """
    from ripser import ripser
    diagrams = ripser(x, maxdim=maxdim)['dgms']
    life = {}
    for dim in range(1, maxdim + 1):
        d = diagrams[dim]
        lt = d[:, 1] - d[:, 0]
        life[dim] = np.sort(lt[np.isfinite(lt)])[::-1]
    return life, diagrams


def kth_longest(life, dim, k):
    """k-th longest lifetime (k = 0 is the longest) in dimension dim, or 0 if there are fewer features."""
    arr = life.get(dim, np.array([]))
    return arr[k] if len(arr) > k else 0.


def matched_gaussian(x, rng):
    """Gaussian points with the same number of points and the same PCA spectrum as x."""
    centered = x - x.mean(0)
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    stds = s / np.sqrt(len(x))
    return (rng.standard_normal((len(x), len(stds))) * stds) @ Vt


def null_lifetimes(x, n_null, rng, features, maxdim=2):
    """
    Lifetimes of selected features in n_null matched Gaussian clouds.

    features: list of (dim, k) pairs, e.g. [(1, 0), (1, 1), (2, 0)] for the
        longest H1, second-longest H1 and longest H2.
    Returns an n_null x len(features) array.
    """
    out = np.zeros((n_null, len(features)))
    for n in range(n_null):
        life, _ = lifetimes(normalize(matched_gaussian(x, rng)), maxdim=maxdim)
        out[n] = [kth_longest(life, dim, k) for dim, k in features]
    return out


def participation_ratio(x):
    """(sum of PCA variances)^2 / sum of squared variances."""
    variances = np.linalg.svd(x - x.mean(0), compute_uv=False)**2
    return variances.sum()**2 / np.sum(variances**2)


def two_nn_dimension(x):
    """TwoNN intrinsic dimension (Facco et al. 2017): MLE from ratios of 2nd to 1st neighbour distances."""
    dist = np.sqrt(((x[:, None, :] - x[None, :, :])**2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    nearest = np.sort(dist, 1)[:, :2]
    mu = nearest[:, 1] / nearest[:, 0]
    mu = mu[np.isfinite(mu) & (mu > 1)]
    return len(mu) / np.sum(np.log(mu))
