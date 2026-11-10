"""
Shape of point clouds (e.g. one cell type's embeddings) without a target shape:
persistent homology and intrinsic dimension.

  normalize            center and scale to unit root-mean-square radius
  lifetimes            lifetimes of H1, H2, ... features (Vietoris-Rips, ripser)
  matched_gaussian     Gaussian cloud with the same size and PCA spectrum
  null_lifetimes       lifetimes of the longest features in matched Gaussian clouds
  two_nn_dimension     TwoNN intrinsic dimension (Facco et al. 2017)
  gride_dimension      Gride intrinsic dimension at a given neighbour scale
                       (Denti et al. 2022); TwoNN is the smallest scale

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


def two_nn_dimension(x):
    """TwoNN intrinsic dimension (Facco et al. 2017): MLE from ratios of 2nd to 1st neighbour distances."""
    dist = np.sqrt(((x[:, None, :] - x[None, :, :])**2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    nearest = np.sort(dist, 1)[:, :2]
    mu = nearest[:, 1] / nearest[:, 0]
    mu = mu[np.isfinite(mu) & (mu > 1)]
    return len(mu) / np.sum(np.log(mu))


def gride_dimension(x, n1):
    """
    Gride intrinsic dimension (Denti et al. 2022) at neighbour scale n1.

    Maximum-likelihood estimate from mu = r_n2 / r_n1, the ratio of the distances from
    each point to its n2-th and n1-th nearest neighbours, with n2 = 2 * n1. For points
    with locally uniform density in d dimensions, mu has density
        f(mu) = d (mu^d - 1)^(n2 - n1 - 1) / (B(n2 - n1, n1) mu^((n2 - 1) d + 1)).
    n1 = 1 is TwoNN. Larger n1 measures the dimension at a larger scale and is less
    sensitive to how regularly the nearest neighbours are spaced.
    Needs n2 < number of points; returns NaN otherwise.
    """
    from scipy.optimize import minimize_scalar
    n2 = 2 * n1
    if n2 >= len(x):
        return np.nan
    dist = np.sqrt(((x[:, None, :] - x[None, :, :])**2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    nearest = np.sort(dist, 1)
    mu = nearest[:, n2 - 1] / nearest[:, n1 - 1]
    log_mu = np.log(mu[np.isfinite(mu) & (mu > 1)])

    def neg_log_likelihood(d):
        # log(mu^d - 1), computed stably; the Beta-function term does not depend on d
        log_mu_d_minus_1 = d * log_mu + np.log1p(-np.exp(-d * log_mu))
        return -np.sum(np.log(d) + (n2 - n1 - 1) * log_mu_d_minus_1
                       - ((n2 - 1) * d + 1) * log_mu)

    return minimize_scalar(neg_log_likelihood, bounds=(1e-3, 100), method='bounded').x
