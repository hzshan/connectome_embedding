from . import utils
import torch
import numpy as np
import tqdm


def init_uniform_param(shape, scale):
    return torch.tensor(
        np.ones(shape) * scale, dtype=torch.float32, requires_grad=True)


class Model:
    """
    Base class for models.
    """
    def __init__(self, N, Ntype, D, onehot_types, wrapper_fn='exp'):
        """
        Args:
        N: number of neurons
        Ntype: number of cell types
        D: embedding dimension
        """
        self.embeddings = torch.tensor(
            np.random.randn(N, D) / np.sqrt(D),
            requires_grad=True, dtype=torch.float32)

        self.A = init_uniform_param((Ntype, Ntype), 1.)
        self.B = init_uniform_param((Ntype, Ntype), 1.)
        self.C = init_uniform_param((Ntype, Ntype), 1.)

        self.N = N
        self.Ntype = Ntype
        self.onehot_types = onehot_types

        assert onehot_types.size() == (N, Ntype)

        self.params = [self.embeddings, self.A, self.B, self.C]

        assert wrapper_fn in ['exp', 'softplus']
        self.wrapper_fn_choice = wrapper_fn


    def expand_type_pairs(self, X):
        """
        Ntype x Ntype matrix of per-type-pair values -> N x N matrix whose (i, j)
        entry is X[type of i, type of j]. Same as onehot @ X @ onehot.T, done by
        indexing.
        """
        types = self.onehot_types.argmax(1)
        return X[types][:, types]


    def wrapper_fn(self, x):
        """A nonlinear function to keep the predicted synaptic number positive."""
        if self.wrapper_fn_choice == 'exp':
            return torch.exp(x)
        else:
            return torch.nn.functional.softplus(x)


    def pred_synapses(self):
        """
        Predict number synapses from stored embeddings.
        Args:
        ctype_onehots: N x Ntype
        """

        eps_dist = 0.01

        scaling = self.expand_type_pairs(self.A)
        covar = self.expand_type_pairs(self.C**2 + eps_dist)
        bias = self.expand_type_pairs(self.B)

        content = scaling * torch.exp(-torch.cdist(
            self.embeddings, self.embeddings)**2 / covar) + bias

        mean = self.wrapper_fn(content)

        return mean - torch.diag(mean.diag())  # remove self-connections


class AsymModel(Model):
    def __init__(self, N, Ntype, D, wrapper_fn='exp'):
        super().__init__(N, Ntype, D, wrapper_fn)


        self.asym_embeddings = torch.tensor(
            np.random.randn(N, D) / np.sqrt(D),
            requires_grad=True, dtype=torch.float32)

        self.A_asym = init_uniform_param((Ntype, Ntype), 1.)
        self.C_asym = init_uniform_param((Ntype, Ntype), 1.)

        self.params += [self.asym_embeddings, self.A_asym, self.C_asym]
        self.asym_mat = torch.sign(-torch.arange(N).view(-1, 1) + torch.arange(N).view(1, -1))

    def pred_synapses(self, ctype_onehots, split=False):
        """
        Predict number synapses from stored embeddings.
        Args:
        ctype_onehots: N x Ntype
        """

        eps_dist = 0.01

        scaling = ctype_onehots @ self.A @ ctype_onehots.T
        scaling_asym = ctype_onehots @ self.A_asym @ ctype_onehots.T

        covar = ctype_onehots @ (self.C**2 + eps_dist) @ ctype_onehots.T
        covar_asym = ctype_onehots @ (self.C_asym**2 + eps_dist) @ ctype_onehots.T

        bias = ctype_onehots @ self.B @ ctype_onehots.T

        sym_part = scaling * torch.exp(-torch.cdist(
            self.embeddings, self.embeddings)**2 / covar)
        asym_part = scaling_asym * torch.exp(-torch.cdist(
            self.asym_embeddings, self.asym_embeddings)**2 / covar_asym)

        # return  torch.exp(scaling * torch.exp(-sq_dist_mat / covar) + bias)

        if split:
            return sym_part, self.asym_mat * asym_part + bias
        else:
            return  torch.relu(sym_part + self.asym_mat * asym_part + bias)


class SourceTargetModel(Model):
    """
    Model with separate source (presynaptic) and target (postsynaptic)
    embeddings, allowing asymmetric expected connectivity (W_ij != W_ji),
    including within a cell type.

    source embedding of neuron i: z_i
    target embedding of neuron j: z_j + m_j   (m = target_modifier)

    E[W_ij] = f(A_cd * exp(-||z_i - (z_j + m_j)||^2 / (C_cd^2 + eps)) + B_cd)

    The modifier is initialized at zero, so training starts from the
    symmetric model; it nests the symmetric model (m = 0), which makes
    likelihood comparisons between the two straightforward. Use
    `modifier_penalty` in a reg_fn to control how far it departs from
    symmetry.
    """

    def __init__(self, N, Ntype, D, onehot_types, wrapper_fn='exp'):
        super().__init__(N, Ntype, D, onehot_types, wrapper_fn=wrapper_fn)

        self.target_modifier = torch.zeros(
            (N, D), requires_grad=True, dtype=torch.float32)

        self.params += [self.target_modifier]

    @property
    def source_embeddings(self):
        return self.embeddings

    @property
    def target_embeddings(self):
        return self.embeddings + self.target_modifier

    def modifier_penalty(self):
        """Mean squared size of the target modifier (for use in reg_fn)."""
        return torch.mean(self.target_modifier**2)

    def pred_synapses(self):
        """
        Predict the mean number of synapses from each neuron (row, source)
        to each neuron (column, target). Self-connections are set to zero.
        """
        eps_dist = 0.01

        scaling = self.expand_type_pairs(self.A)
        covar = self.expand_type_pairs(self.C**2 + eps_dist)
        bias = self.expand_type_pairs(self.B)

        dist_sq = torch.cdist(self.source_embeddings,
                              self.target_embeddings)**2

        mean = self.wrapper_fn(scaling * torch.exp(-dist_sq / covar) + bias)

        return mean - torch.diag(mean.diag())  # remove self-connections


class InteractionModel(Model):
    """
    Model with a rotation matrix for each pair of cell types.
    """
    def __init__(self, N, Ntype, D, onehot_types, wrapper_fn='exp', kernel='distance', nonlinear_fn='exp'):
        super().__init__(N, Ntype, D, onehot_types,
                         wrapper_fn=wrapper_fn)

        # there is a separate DxD rotation matrix for each pair of cell types,
        # and a D-dim translation vector
        self.rotation_params = torch.tensor(
            np.zeros((Ntype, Ntype, D, D)),
            requires_grad=True, dtype=torch.float32)

        self.translation_vecs = torch.tensor(
            np.zeros((Ntype, Ntype, D)),
            requires_grad=True, dtype=torch.float32)
        
        self.params += [self.rotation_params,
                        self.translation_vecs]
        
        self.use_transforms = True  # if False, use the original embeddings

        assert kernel in ['distance', 'dot_product']
        self.kernel_type = kernel
        self.nonlinear_fn = nonlinear_fn
        


    def get_target_embs_for_plots(self, ctype_onehots, source_type):
        _, _Ntype = ctype_onehots.size()

        # Ntype x D x D
        rotation_mats = utils.make_rotation_mats(
            self.rotation_params[source_type])

        all_target_embs = []
        for target_type in range(_Ntype):
            target_inds = ctype_onehots[:, target_type].bool()
            all_target_embs.append(
                self.embeddings[target_inds] @\
                        rotation_mats[target_type] +\
                            self.translation_vecs[source_type, target_type])
        return torch.cat(all_target_embs, 0).detach()  # N x D


    def get_source_embs_for_plots(self, ctype_onehots, target_type):
        _, _Ntype = ctype_onehots.size()

        # Ntype x D x D
        rotation_mats = utils.make_rotation_mats(
            self.rotation_params[:, target_type])

        all_source_embs = []
        for source_type in range(_Ntype):
            source_inds = ctype_onehots[:, source_type].bool()
            all_source_embs.append(
                (self.embeddings[source_inds] - self.translation_vecs[source_type, target_type]) @\
                        rotation_mats[source_type].T)
        return torch.cat(all_source_embs, 0).detach()  # N x D
        
    def get_rotation_anti_sym(self):
        return self.rotation_params - self.rotation_params.transpose(-1, -2)

    def get_rotation_mats(self):
        return utils.make_rotation_mats(self.rotation_params)


    # runs uncompiled inside train_model(compile=True): the loop over types
    # does not compile
    @torch.compiler.disable
    def get_dist_mat_with_transforms(self):
        """
        Squared distance between every source neuron i and target neuron j,
        after rotating the target's embedding by the rotation for their pair of
        cell types: d_ij = || z_j R_{t_i t_j} - z_i ||^2. Returns N x N.
        """
        # Rotation matrices for every pair of cell types, indexed
        # [source type, target type]: Ntype x Ntype x D x D.
        rotation_mats = self.get_rotation_mats()

        # Cell type of each neuron as an integer label (0 .. Ntype-1): N
        types = self.onehot_types.argmax(1)

        # Expanding the square splits the distance into three terms:
        #     d_ij = ||z_i||^2 + ||z_j R||^2 - 2 z_i . (z_j R)
        # Because R preserves length, ||z_j R||^2 = ||z_j||^2, so the first two
        # terms don't involve the rotation at all. Compute each neuron's squared
        # length once here: N
        sq_norms = (self.embeddings**2).sum(1)

        # Only the cross term z_i . (z_j R) needs the rotation. Writing it as
        # z_j . (R z_i) means we can rotate the SOURCE neuron instead: for a fixed
        # target type, each source neuron needs only one rotation, however many
        # neurons that target type has. So we go one target type at a time.
        cross_blocks = []      # one N x n_d block of cross terms per target type
        target_inds_all = []   # which neurons (columns) each block covers

        for target_type in range(self.Ntype):
            # Indices of the neurons that belong to this target type
            target_inds = torch.nonzero(self.onehot_types[:, target_type], as_tuple=True)[0]

            # A cell type with no neurons contributes no columns, so skip it
            if len(target_inds) == 0:
                continue

            # For every source neuron i, look up the rotation R_{t_i, target_type}
            # that applies when it connects to this target type: N x D x D.
            # This is a lookup by type, so it stays small (one D x D per neuron).
            source_rotations = rotation_mats[types, target_type]

            # Rotate each source neuron once: R z_i for every i: N x D
            rotated_sources = torch.einsum('ixy,iy->ix', source_rotations, self.embeddings)

            # Cross term z_j . (R z_i) for all sources i and every target j of this
            # type, done as one matrix multiply: N x n_d
            cross_blocks.append(rotated_sources @ self.embeddings[target_inds].T)

            # Remember which neurons these columns belong to
            target_inds_all.append(target_inds)

        # perm[k] = the neuron that column k of the stacked blocks belongs to.
        # The blocks are grouped by type, so this is not the original neuron order.
        perm = torch.cat(target_inds_all)

        # inv_perm[j] = the column where neuron j ended up, i.e. the inverse of perm
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(len(perm), device=perm.device)

        # Stack the blocks side by side, then reorder the columns so that
        # column j is neuron j again: N x N
        cross = torch.cat(cross_blocks, 1)[:, inv_perm]

        # Add the three terms back together. sq_norms[:, None] varies along rows
        # (the source's length), sq_norms[None, :] along columns (the target's).
        # Computed this way, the largest array is N x N; building a separate
        # rotation for every neuron pair would need an N x N x D x D tensor.
        return sq_norms[:, None] + sq_norms[None, :] - 2 * cross


    def pred_synapses(self, pairs=None):
        """
        Predicted mean number of synapses.

        With no arguments: the N x N matrix, entry (i, j) for synapses from
        neuron i to neuron j.
        With pairs, a K x 2 integer tensor of neuron indices, row k = (source,
        target): one prediction per pair, for synapses from neuron pairs[k, 0] to
        neuron pairs[k, 1]; the same values as the matrix entries at those
        pairs, computed without forming the matrix.
        """
        eps_dist = 0.01
        by_pair = pairs is not None

        if by_pair:
            source_inds, target_inds = pairs[:, 0], pairs[:, 1]
            # kernel parameters of each pair's (source type, target type)
            types = self.onehot_types.argmax(1)
            source_types, target_types = types[source_inds], types[target_inds]
            scaling = self.A[source_types, target_types]
            covar = self.C[source_types, target_types]**2 + eps_dist
            bias = self.B[source_types, target_types]
            z_source, z_target = self.embeddings[source_inds], self.embeddings[target_inds]
        else:
            scaling = self.expand_type_pairs(self.A)
            covar = self.expand_type_pairs(self.C**2 + eps_dist)
            bias = self.expand_type_pairs(self.B)

        if self.use_transforms:
            if self.kernel_type == 'dot_product':
                raise NotImplementedError(
                    "dot_product kernel is not implemented when use_transforms=True"
                )
            if by_pair:
                # as get_dist_mat_with_transforms, one rotation per pair:
                # d = ||z_i||^2 + ||z_j||^2 - 2 z_j . (R_{t_i t_j} z_i)
                rotations = self.get_rotation_mats()[source_types, target_types]
                rotated_sources = torch.einsum('kxy,ky->kx', rotations, z_source)
                dist_sq = ((z_source**2).sum(1) + (z_target**2).sum(1)
                           - 2 * (rotated_sources * z_target).sum(1))
            else:
                dist_sq = self.get_dist_mat_with_transforms()
        elif self.kernel_type == 'distance':
            if by_pair:
                dist_sq = ((z_source - z_target)**2).sum(1)
            else:
                dist_sq = torch.cdist(self.embeddings, self.embeddings)**2
        else:
            assert self.kernel_type == 'dot_product'
            if by_pair:
                dist_sq = -(z_source * z_target).sum(1)
            else:
                dist_sq = -self.embeddings @ self.embeddings.T

        nonlinear_fn = None
        if self.nonlinear_fn == 'exp':
            nonlinear_fn = lambda x: torch.exp(x)
        elif self.nonlinear_fn == 'cubic':
            nonlinear_fn = lambda x: x**3
        elif self.nonlinear_fn == 'linear':
            nonlinear_fn = lambda x: x
        else:
            raise ValueError(f'unknown nonlinear_fn {self.nonlinear_fn}')

        content = scaling * nonlinear_fn(-dist_sq / covar) + bias

        return self.wrapper_fn(content)


class PairSampler:
    """
    Draws neuron pairs for a stochastic estimate of the mean loss over the
    pairs in a mask (the loss of masked_loss), without forming N x N
    predictions.

    The pairs in the mask are split into connected pairs (target > 0) and
    unconnected pairs (target = 0). Each draw takes n_connected pairs from the
    first group and n_unconnected from the second, uniformly with replacement;
    the estimate weights each group's mean loss by the group's share of the
    mask's pairs, so it is unbiased and has the scale of the full mean loss.
    Sampling the sparse connected pairs separately keeps the variance low.

    target_mat: N x N array of synapse counts (dense, or a scipy.sparse matrix)
    mask: N x N boolean array of the pairs in the loss, or None for all
        off-diagonal pairs
    """
    def __init__(self, target_mat, mask, n_connected, n_unconnected, seed=0):
        import scipy.sparse as sp
        self.N = target_mat.shape[0]
        W = sp.coo_matrix(target_mat)
        rows, cols, vals = W.row, W.col, W.data.astype(np.float32)
        keep = (vals > 0) & (rows != cols)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            assert mask.shape == (self.N, self.N)
            keep &= mask[rows, cols]
        self.rows = torch.as_tensor(rows[keep], dtype=torch.long)
        self.cols = torch.as_tensor(cols[keep], dtype=torch.long)
        self.vals = torch.as_tensor(vals[keep])
        self.mask = mask
        # sorted linear indices of the connected pairs, to reject them when
        # drawing unconnected pairs
        self.connected_index = torch.sort(self.rows * self.N + self.cols).values

        n_pairs = (self.N * (self.N - 1) if mask is None
                   else int(mask.sum() - np.diag(mask).sum()))
        self.n_connected_total = len(self.vals)
        self.n_unconnected_total = n_pairs - self.n_connected_total
        self.weight_connected = self.n_connected_total / n_pairs
        self.weight_unconnected = self.n_unconnected_total / n_pairs
        self.n_connected = min(n_connected, self.n_connected_total)
        self.n_unconnected = n_unconnected
        self.generator = torch.Generator().manual_seed(seed)

    def _unconnected(self):
        """n_unconnected pairs drawn uniformly from the unconnected pairs in the mask.
        This does not produce the full list of unconnected pairs and then sample;
        instead, it samples from all pairs and throws away connected pairs"""
        found = []
        n = 0   # accepted pairs so far
        while n < self.n_unconnected:
            # draw 20% more pairs than still needed, to allow for rejections
            m = int(1.2 * (self.n_unconnected - n)) + 16
            # random (source, target) pairs, uniform over all ordered pairs
            i = torch.randint(self.N, (m,), generator=self.generator)
            j = torch.randint(self.N, (m,), generator=self.generator)
            # one number per pair, in the same encoding as connected_index
            index = i * self.N + j
            # binary search; clamp since an index above all connected ones lands past the end
            pos = torch.searchsorted(self.connected_index, index).clamp(max=len(self.connected_index) - 1)
            # accept: not a self-connection, and not found in connected_index
            ok = (i != j) & (self.connected_index[pos] != index)
            if self.mask is not None:
                # also require the pair to be in the mask
                ok &= torch.as_tensor(self.mask[i.numpy(), j.numpy()])
            found.append(torch.stack([i[ok], j[ok]], 1)); n += int(ok.sum())
        # the last pass may overshoot; keep exactly n_unconnected
        return torch.cat(found)[:self.n_unconnected]

    def sample(self):
        """(pairs, target values, weights), pairs a K x 2 tensor of (source,
        target) neuron indices: weighting each pair's
        loss by its weight and summing gives the loss estimate."""
        k = torch.randint(self.n_connected_total, (self.n_connected,), generator=self.generator)
        pairs = torch.cat([torch.stack([self.rows[k], self.cols[k]], 1), self._unconnected()])
        values = torch.cat([self.vals[k], torch.zeros(self.n_unconnected)])
        weights = torch.cat([
            torch.full((self.n_connected,), self.weight_connected / self.n_connected),
            torch.full((self.n_unconnected,), self.weight_unconnected / self.n_unconnected)])
        return pairs, values, weights


def pair_loss(preds, target_values, weights, loss_type):
    """Weighted sum of the per-pair loss of masked_loss (see PairSampler)."""
    if loss_type == 'poisson':
        elementwise = preds - target_values * torch.log(preds + 1e-8)
    else:
        assert loss_type == 'mse'
        elementwise = (preds - target_values)**2 / (1 + target_values**2)
    return (elementwise * weights).sum()


def masked_loss(preds, target_mat, mask, loss_type):
    """
    Mean loss over the entries of target_mat where mask is 1.
    poisson: Poisson negative log-likelihood up to a constant,
        preds - target * log(preds + 1e-8) (as torch.nn.PoissonNLLLoss with
        log_input=False).
    mse: squared error weighted by 1 / (1 + target^2).
    """
    if loss_type == 'poisson':
        elementwise = preds - target_mat * torch.log(preds + 1e-8)
    else:
        assert loss_type == 'mse'
        elementwise = (preds - target_mat)**2 / (1 + target_mat**2)
    # summed in float64: a float32 sum over N^2 entries loses precision for large N
    return ((elementwise * mask).sum(dtype=torch.float64) / mask.sum()).float()


def train_model(model, target_mat,
                loss_type='poisson', lr=0.01, steps=40000,
                print_every=2000, reg_fn=None, train_mask=None, test_mask=None,
                compile=True, pairs_per_step=None, pair_seed=0, lr_end=None):
    """
    Train an embedding model to fit the connectivity matrix with Adam.

    Args:
    model: a Model; its pred_synapses() returns the N x N predicted means
    target_mat: N x N tensor of synapse counts (with pairs_per_step, also a
        scipy.sparse matrix)
    loss_type: 'poisson' or 'mse' (see masked_loss)
    reg_fn: optional function of the model, added to the loss
    train_mask: N x N boolean array, entries used for the loss. Default: all
        off-diagonal entries.
    test_mask: N x N boolean array; if given, the loss on these entries is
        printed every print_every steps.
    compile: compile the loss computation with torch.compile, which fuses the
        operations on N x N matrices (several times faster for large N; the
        first step takes longer while it compiles). Falls back to uncompiled
        if compilation fails.
    pairs_per_step: None for the loss over all pairs at every step. Otherwise
        (n_connected, n_unconnected): each step uses an unbiased estimate of
        that loss from n_connected connected and n_unconnected unconnected
        neuron pairs drawn at random (PairSampler), so memory and time per step
        do not grow as N^2. Needs a model whose pred_synapses takes
        a K x 2 tensor of neuron pairs (InteractionModel).
        The losses returned are the estimates.
    pair_seed: seed of the pair sampling.
    lr_end: None for a constant learning rate. Otherwise the learning rate decays
        exponentially from lr at the first step to lr_end at the last, which
        lets the noisy steps of pair sampling settle.

    Returns: losses (training loss without reg_fn, per step),
    norms of the embeddings per step, and an array of zeros (unused).
    """
    _N = target_mat.shape[0]

    if pairs_per_step is not None:
        n_connected, n_unconnected = pairs_per_step
        sampler = PairSampler(target_mat, train_mask, n_connected, n_unconnected, seed=pair_seed)
        test_sampler = (None if test_mask is None else
                        PairSampler(target_mat, test_mask, n_connected, n_unconnected, seed=pair_seed + 1))

        def pair_estimate(s):
            pairs, values, weights = s.sample()
            return pair_loss(model.pred_synapses(pairs), values, weights, loss_type)

        def total_loss():
            loss = pair_estimate(sampler)
            tot_loss = loss + reg_fn(model) if reg_fn is not None else loss
            return tot_loss, loss, None
        loss_fn = total_loss
    else:
        target_mat = torch.as_tensor(target_mat, dtype=torch.float32)

        if train_mask is None:
            train_mask = ~np.eye(_N, dtype=bool)
        else:
            assert train_mask.shape == target_mat.shape
            assert train_mask.dtype == bool
        train_mask = torch.as_tensor(train_mask, dtype=torch.float32)

        if test_mask is not None:
            assert test_mask.shape == target_mat.shape
            assert test_mask.dtype == bool
            test_mask = torch.as_tensor(test_mask, dtype=torch.float32)

        def total_loss():
            preds = model.pred_synapses()
            loss = masked_loss(preds, target_mat, train_mask, loss_type)
            tot_loss = loss + reg_fn(model) if reg_fn is not None else loss
            return tot_loss, loss, preds

        loss_fn = total_loss
        if compile:
            try:
                # compile the forward and backward passes now, so that a failure
                # falls back to uncompiled training here
                compiled = torch.compile(total_loss)
                compiled()[0].backward()
                loss_fn = compiled
            except Exception as err:
                print(f'torch.compile failed, training uncompiled: {type(err).__name__}')

    optim = torch.optim.Adam(params=model.params, lr=lr)
    scheduler = (None if lr_end is None else
                 torch.optim.lr_scheduler.ExponentialLR(optim, gamma=(lr_end / lr) ** (1 / max(steps - 1, 1))))

    losses = np.zeros(steps)
    embedding_norms = np.zeros(steps)
    rotation_norms = np.zeros(steps)

    for i in tqdm.trange(steps):
        optim.zero_grad()
        tot_loss, loss, preds = loss_fn()

        # a non-finite prediction makes the loss non-finite
        if not torch.isfinite(tot_loss):
            raise ValueError(f'loss is {tot_loss.item()} at step {i}')

        tot_loss.backward()
        optim.step()
        if scheduler is not None:
            scheduler.step()

        losses[i] = loss.item()
        embedding_norms[i] = torch.norm(model.embeddings).item()

        if (i+1) % print_every == 0:
            for p in model.params:
                if not torch.isfinite(p).all():
                    raise ValueError(f'non-finite parameter at step {i}')
            print(i, f'normalized loss: {losses[i]:.4f}',
                f'avg sq embed norm: {embedding_norms[i]**3 / _N:.2f}')
            if test_mask is not None:
                with torch.no_grad():
                    if pairs_per_step is not None:
                        test_loss = pair_estimate(test_sampler)
                    else:
                        test_loss = masked_loss(preds, target_mat, test_mask, loss_type)
                print(f'test loss: {test_loss:.4f}')

    return losses, embedding_norms, rotation_norms
