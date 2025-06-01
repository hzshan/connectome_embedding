import utils
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


    def wrapper_fn(self, x):
        """A nonlinear function to keep the predicted synaptic number positive."""
        if self.wrapper_fn == 'exp':
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

        scaling = self.onehot_types @ self.A @ self.onehot_types.T
        covar = self.onehot_types @ (self.C**2 + eps_dist) @ self.onehot_types.T
        bias = self.onehot_types @ self.B @ self.onehot_types.T

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
    Model with separate source and target embeddings.

    target embeddings = embeddings + target_modifier
    """

    def __init__(self, N, Ntype, D, wrapper_fn='exp'):
        super().__init__(N, Ntype, D, wrapper_fn)

        self.target_modifier = torch.tensor(
            np.zeros((N, D)),
            requires_grad=True,
            dtype=torch.float32)
        
        self.params += [self.target_modifier]
    
    def pred_synapses(self, ctype_onehots):
        """
        ctype_onehots: N x Ntype
        """
        eps_dist = 0.01

        scaling = ctype_onehots @ self.A @ ctype_onehots.T
        covar = ctype_onehots @ (self.C**2 + eps_dist) @ ctype_onehots.T
        bias = ctype_onehots @ self.B @ ctype_onehots.T

        source_emb = self.embeddings
        target_emb = self.embeddings + self.target_modifier

        dist_sq = torch.cdist(source_emb, target_emb)**2

        content = scaling * torch.exp(-dist_sq / covar) + bias

        return  self.wrapper_fn(content)
    

class InteractionModel(Model):
    """
    Model with a rotation matrix for each pair of cell types.
    """
    def __init__(self, N, Ntype, D, onehot_types, wrapper_fn='exp'):
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


    def get_dist_mat_with_transforms(self):
        rotation_mats = self.get_rotation_mats()

        # einsum notations: i, j index cells; c, d index cell types; x, y index embedding dimensions
        blown_rot_mats = torch.einsum('ic,cdxy->idxy',
                                      self.onehot_types, rotation_mats)
        blown_rot_mats = torch.einsum('idxy,jd->ijxy',
                                      blown_rot_mats, self.onehot_types) # N_source x N_target x D x D

        transformed_target_embs = torch.einsum('jx,ijxy->ijy', self.embeddings, blown_rot_mats) # N_source x N_target x D
        expanded_source_embs = torch.einsum('ix,j->ijx', self.embeddings, torch.ones(self.N)) # N_source x N_target x D

        distance = torch.norm(transformed_target_embs - expanded_source_embs, dim=2)**2
        return distance


    def pred_synapses(self):
        """

        """
        eps_dist = 0.01

        scaling = self.onehot_types @ self.A @ self.onehot_types.T
        covar = self.onehot_types @ (self.C**2 + eps_dist) @ self.onehot_types.T
        bias = self.onehot_types @ self.B @ self.onehot_types.T

        # emb_sum = torch.sum(source_emb**2, 1, keepdim=True)
        # target_sum = torch.sum(target_emb**2, 1, keepdim=True)
        # cross_term = source_emb @ target_emb.T
        # dist = emb_sum + target_sum.T - 2 * cross_term
        if self.use_transforms:
            dist_sq = self.get_dist_mat_with_transforms()
        else:
            dist_sq = torch.cdist(self.embeddings, self.embeddings)**2

        content = scaling * torch.exp(-dist_sq / covar) + bias
        # content = -dist * scaling + bias

        return self.wrapper_fn(content)

def train_model(model, target_mat,
                loss_type='poisson', lr=0.01, steps=40000,
                print_every=2000, reg_fn=None, train_inds=None, test_inds=None):
    """
    Train an embedding model to fit the connectivity matrix.

    """

    # only compute loss on off-diagonal elements of J
    _N = target_mat.shape[0]

    if train_inds is None:
        train_inds = np.where((np.ones([_N, _N]) - np.diag(np.ones(_N))).flatten())[0]
    y_train = target_mat.flatten()[train_inds]


    if test_inds is not None:
        y_test = target_mat.flatten()[test_inds]
    else:
        y_test = None

    optim = torch.optim.Adam(params=model.params, lr=lr)

    losses = np.zeros(steps)
    embedding_norms = np.zeros(steps)
    rotation_norms = np.zeros(steps)

    poisson_loss = torch.nn.PoissonNLLLoss(log_input=False)


    for i in tqdm.trange(steps):
        optim.zero_grad()

        flat_preds = model.pred_synapses().flatten()

        if loss_type == 'poisson':
            loss = poisson_loss(flat_preds[train_inds], y_train)
        else:
            assert loss_type == 'mse'
            # loss = -torch.dot(2 * torch.sigmoid(flat_preds[train_inds])-1, y_train) -\
            #       torch.dot(2-2*torch.sigmoid(flat_preds[train_inds]), 1 - y_train)
            # torch.binary_cross_entropy_with_logits(
            #     , reduction=1)
            # loss = torch.norm(flat_preds[train_inds] - y_train)**2
            loss = torch.sum((flat_preds[train_inds] - y_train)**2 / (1 + flat_preds[train_inds])**2)
            # loss = torch.sum((torch.log(flat_preds[train_inds] + 1e-1) - torch.log(y_train + 1e-1))**2)


        if reg_fn is not None:
            tot_loss = loss + reg_fn(model)
        else:
            tot_loss = loss

        tot_loss.backward()

        if torch.isnan(tot_loss):
            raise ValueError('loss is nan')
        optim.step()

        losses[i] = loss.detach().numpy()
        embedding_norms[i] = torch.norm(model.embeddings).detach().numpy()

        # if model.rotation_params is not None:
        #     rotation_norms[i] = torch.norm(model.rotation_params).detach().numpy()

        if loss_type == 'mse':
            losses[i] /= (torch.norm(y_train)**2).numpy()

        if (i+1) % print_every == 0:
            print(i, f'normalized loss: {losses[i]:.4f}',
                f'avg sq embed norm: {torch.norm(model.embeddings).detach().numpy()**2 / _N:.2f}')
            if y_test is not None:
                test_loss = poisson_loss(flat_preds[test_inds], y_test)
                print(f'test loss: {test_loss:.4f}')
    
    return losses, embedding_norms, rotation_norms