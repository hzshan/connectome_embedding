import utils
import torch
import numpy as np
import pandas

def get_squared_dist(source_emb, target_emb):
    """
    Assuming a NxD shape, return a NxN matrix of squared distances."""
    source_sum_sq = torch.sum(source_emb * source_emb, 1, keepdim=True)
    target_sum_sq = torch.sum(target_emb * target_emb, 1, keepdim=True)

    return source_sum_sq + target_sum_sq.T - 2 * (source_emb @ target_emb.T)


def make_rotation_mats(rotation_params):
    """
    rotation_params: (Ntype, D, D)

    Recover SO(D) matrices via matrix exponential
    """
    anti_sym = rotation_params - rotation_params.transpose(-1, -2)

    return torch.matrix_exp(anti_sym)


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

        self.A = utils.init_uniform_param((Ntype, Ntype), 1.)
        self.B = utils.init_uniform_param((Ntype, Ntype), 1.)
        self.C = utils.init_uniform_param((Ntype, Ntype), 1.)

        self.N = N
        self.Ntype = Ntype
        self.onehot_types = onehot_types

        assert onehot_types.size() == (N, Ntype)

        self.params = [self.embeddings, self.A, self.B, self.C]

        assert wrapper_fn in ['exp', 'relu']
        self.wrapper_fn = 'exp'


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

        content = scaling * torch.exp(-get_squared_dist(self.embeddings, self.embeddings) / covar) + bias

        return  torch.exp(content) if self.wrapper_fn == 'exp' else torch.relu(content)


class AsymModel(Model):
    def __init__(self, N, Ntype, D, wrapper_fn='exp'):
        super().__init__(N, Ntype, D, wrapper_fn)


        self.asym_embeddings = torch.tensor(
            np.random.randn(N, D) / np.sqrt(D),
            requires_grad=True, dtype=torch.float32)

        self.A_asym = utils.init_uniform_param((Ntype, Ntype), 1.)
        self.C_asym = utils.init_uniform_param((Ntype, Ntype), 1.)

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

        sym_part = scaling * torch.exp(-get_squared_dist(self.embeddings) / covar)
        asym_part = scaling_asym * torch.exp(-get_squared_dist(self.asym_embeddings) / covar_asym)

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

        dist = get_squared_dist(source_emb, target_emb)

        content = scaling * torch.exp(-dist / covar) + bias

        return  torch.exp(content) if self.wrapper_fn == 'exp' else torch.relu(content)
    

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
        rotation_mats = make_rotation_mats(self.rotation_params[source_type])

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
        rotation_mats = make_rotation_mats(self.rotation_params[:, target_type])

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
        return make_rotation_mats(self.rotation_params)
    
    # def get_dist_mat(self, ctype_onehots):
    #     rotation_mats = self.get_rotation_mats()

    #     _N, _Ntype = ctype_onehots.size()
    #     dist = torch.zeros((_N, _N))

    #     for i in range(_Ntype):
    #         for j in range(_Ntype):
    #             source_inds = torch.nonzero(ctype_onehots[:, i]).squeeze()
    #             target_inds = torch.nonzero(ctype_onehots[:, j]).squeeze()
    #             source_embs = self.embeddings[source_inds]
    #             target_embs = self.embeddings[target_inds] @ rotation_mats[i, j] + self.translation_vecs[i, j]
    #             dist[source_inds[:, None], target_inds] = get_squared_dist(source_embs, target_embs)
    #     return dist


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
            dist = self.get_dist_mat_with_transforms()
        else:
            dist = get_squared_dist(self.embeddings, self.embeddings)

        content = scaling * torch.exp(-dist / covar) + bias

        return  torch.exp(content) if self.wrapper_fn == 'exp' else torch.relu(content)
    