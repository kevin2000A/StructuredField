import torch
import torch.nn as nn
import numpy as np
import math
# import tinycudann as tcnn

tcnn_config = {
	"encoding": {
		"otype": "HashGrid",
		"n_levels": 16,
		"n_features_per_level": 2,
		"log2_hashmap_size": 15,
		"base_resolution": 16,
		"per_level_scale": 1.5
	},
	"network": {
		"otype": "FullyFusedMLP",
		"activation": "ReLU",
		"output_activation": "None",
		"n_neurons": 64,
		"n_hidden_layers": 2
	}
}
 
# Anneal, Coarse-to-Fine Optimization part proposed by:
# Park, Keunhong, et al. Nerfies: Deformable neural radiance fields. CVPR 2021.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()


    def create_embedding_fn(self):
        embed_fns = []
        self.input_dims = self.kwargs['input_dims']
        out_dim = 0
        self.include_input = self.kwargs['include_input']
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += self.input_dims

        max_freq = self.kwargs['max_freq_log2']
        self.num_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, self.num_freqs) * math.pi
        else:
            freq_bands = torch.linspace(2.**0.*math.pi, 2.**max_freq*math.pi, self.num_freqs)

        self.num_fns = len(self.kwargs['periodic_fns'])
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += self.input_dims

        self.embed_fns = embed_fns
        self.out_dim = out_dim


    # Anneal. Initial alpha value is 0, which means it does not use any PE (positional encoding)!
    def embed(self, inputs, alpha_ratio=0.):
        output = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        start = 0
        if self.include_input:
            start = 1
        for i in range(self.num_freqs):
            output[:, (self.num_fns*i+start)*self.input_dims:(self.num_fns*(i+1)+start)*self.input_dims] *= (1.-math.cos(
                math.pi*(max(min(alpha_ratio*self.num_freqs-i, 1.), 0.))
            )) * .5
        return output


def get_embedder(multires, input_dims=3, hash_encoding=False):
    if hash_encoding:
        pass
    else:
        embed_kwargs = {
            'include_input': True,
            'input_dims': input_dims,
            'max_freq_log2': multires-1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, alpha_ratio, eo=embedder_obj): return eo.embed(x, alpha_ratio)
    return embed, embedder_obj.out_dim
    



class MLP(nn.Module):
    def __init__(self, c_in, c_out, c_hiddens, act=nn.LeakyReLU, bn=nn.BatchNorm1d, zero_init=False):
        super().__init__()
        layers = []
        d_in = c_in
        for d_out in c_hiddens:
            layers.append(nn.Linear(d_in, d_out)) # nn.Conv1d(d_in, d_out, 1, 1, 0)
            if bn is not None:
                layers.append(bn(d_out))
            layers.append(act())
            d_in = d_out
        layers.append(nn.Linear(d_in, c_out)) # nn.Conv1d(d_in, c_out, 1, 1, 0)
        if zero_init:
            nn.init.constant_(layers[-1].bias, 0.0)
            nn.init.constant_(layers[-1].weight, 0.0)
        self.mlp = nn.Sequential(*layers)
        self.c_out = c_out


    def forward(self, x):
        return self.mlp(x)


class Shift(nn.Module):
    def __init__(self, shift) -> None:
        super().__init__()
        self.shift = shift


    def forward(self, x):
        return x + self.shift


class BaseProjectionLayer(nn.Module):
    @property
    def proj_dims(self):
        raise NotImplementedError()


    def forward(self, x):
        raise NotImplementedError()


class ProjectionLayer(BaseProjectionLayer):
    def __init__(self, input_dims, proj_dims):
        super().__init__()
        self._proj_dims = proj_dims

        self.proj = nn.Sequential(
            nn.Linear(input_dims, 2 * proj_dims), nn.ReLU(), nn.Linear(2 * proj_dims, proj_dims)
        )


    @property
    def proj_dims(self):
        return self._proj_dims


    def forward(self, x):
        return self.proj(x)


class CouplingLayer(nn.Module):
    def __init__(self, map_s, map_t, projection, mask):
        super().__init__()
        self.map_s = map_s
        self.map_t = map_t
        self.projection = projection
        self.register_buffer("mask", mask) # 1,1,1,3 -> 1,3


    def forward(self, F, y, alpha_ratio):
        y1 = y * self.mask

        F_y1 = torch.cat([F, self.projection(y1,alpha_ratio)], dim=-1)
        s = self.map_s(F_y1)
        t = self.map_t(F_y1)

        x = y1 + (1 - self.mask) * ((y - t) * torch.exp(-s))
        ldj = (-s).sum(-1)

        return x, ldj


    def inverse(self, F, x, alpha_ratio):
        x1 = x * self.mask

        F_x1 = torch.cat([F, self.projection(x1,alpha_ratio)], dim=-1)
        s = self.map_s(F_x1)
        t = self.map_t(F_x1)

        y = x1 + (1 - self.mask) * (x * torch.exp(s) + t)
        ldj = s.sum(-1)

        return y, ldj


def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))


def euler2rot_inv(euler_angle):
    batch_size = euler_angle.shape[0]
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_z, torch.bmm(rot_y, rot_x))


def euler2rot_2d(euler_angle):
    # (B, 1) -> (B, 2, 2)
    theta = euler_angle.reshape(-1, 1, 1)
    rot = torch.cat((
        torch.cat((theta.cos(), theta.sin()), 1),
        torch.cat((-theta.sin(), theta.cos()), 1),
    ), 2)

    return rot


def euler2rot_2dinv(euler_angle):
    # (B, 1) -> (B, 2, 2)
    theta = euler_angle.reshape(-1, 1, 1)
    rot = torch.cat((
        torch.cat((theta.cos(), -theta.sin()), 1),
        torch.cat((theta.sin(), theta.cos()), 1),
    ), 2)

    return rot


def quaternions_to_rotation_matrices(quaternions):
    """
    Arguments:
    ---------
        quaternions: Tensor with size ...x4, where ... denotes any shape of
                     quaternions to be translated to rotation matrices
    Returns:
    -------
        rotation_matrices: Tensor with size ...x3x3, that contains the computed
                           rotation matrices
    """
    # Allocate memory for a Tensor of size ...x3x3 that will hold the rotation
    # matrix along the x-axis
    shape = quaternions.shape[:-1] + (3, 3)
    R = quaternions.new_zeros(shape)

    # A unit quaternion is q = w + xi + yj + zk
    xx = quaternions[..., 1] ** 2
    yy = quaternions[..., 2] ** 2
    zz = quaternions[..., 3] ** 2
    ww = quaternions[..., 0] ** 2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = torch.zeros_like(n)
    s[n != 0] = 2 / n[n != 0]

    xy = s[..., 0] * quaternions[..., 1] * quaternions[..., 2]
    xz = s[..., 0] * quaternions[..., 1] * quaternions[..., 3]
    yz = s[..., 0] * quaternions[..., 2] * quaternions[..., 3]
    xw = s[..., 0] * quaternions[..., 1] * quaternions[..., 0]
    yw = s[..., 0] * quaternions[..., 2] * quaternions[..., 0]
    zw = s[..., 0] * quaternions[..., 3] * quaternions[..., 0]

    xx = s[..., 0] * xx
    yy = s[..., 0] * yy
    zz = s[..., 0] * zz

    R[..., 0, 0] = 1 - yy - zz
    R[..., 0, 1] = xy - zw
    R[..., 0, 2] = xz + yw

    R[..., 1, 0] = xy + zw
    R[..., 1, 1] = 1 - xx - zz
    R[..., 1, 2] = yz - xw

    R[..., 2, 0] = xz - yw
    R[..., 2, 1] = yz + xw
    R[..., 2, 2] = 1 - xx - yy

    return R


# Deform
class DeformNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 scale_included,
                 n_blocks,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 weight_norm=True):
        super(DeformNetwork, self).__init__()
        
        self.n_blocks = n_blocks
        self.skip_in = skip_in
        d_in = 3
        if scale_included:
            d_out_1 = 2
            d_out_2 = 5
        else:
            d_out_2 = 3
            d_out_1 = 1
        # part a
        # xy -> z
        self.scale_included = scale_included
        ori_in = d_in - 1
        dims_in = ori_in
        dims = [dims_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out_1]

        self.embed_fn_1 = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=dims_in)
            self.embed_fn_1 = embed_fn
            dims_in = input_ch
            dims[0] = input_ch + d_feature

        self.num_layers_a = len(dims)
        for i_b in range(self.n_blocks):
            for l in range(0, self.num_layers_a - 1):
                if l + 1 in self.skip_in:
                    out_dim = dims[l + 1] - dims_in - d_feature
                else:
                    out_dim = dims[l + 1]

                lin = nn.Linear(dims[l], out_dim)

                if l == self.num_layers_a - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight, 0.0)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :ori_in], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, ori_in:], 0.0)
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :-(dims_in - ori_in)], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims_in - ori_in):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

                if weight_norm and l < self.num_layers_a - 2:
                    lin = nn.utils.weight_norm(lin)

                setattr(self, "lin"+str(i_b)+"_a_"+str(l), lin)

        # part b
        # z -> xy
        ori_in = 1
        dims_in = ori_in
        dims = [dims_in + d_feature] + [d_hidden] + [d_out_2]

        self.embed_fn_2 = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=dims_in)
            self.embed_fn_2 = embed_fn
            dims_in = input_ch
            dims[0] = input_ch + d_feature

        self.num_layers_b = len(dims)
        for i_b in range(self.n_blocks):
            for l in range(0, self.num_layers_b - 1):
                if l + 1 in self.skip_in:
                    out_dim = dims[l + 1] - dims_in
                else:
                    out_dim = dims[l + 1]

                lin = nn.Linear(dims[l], out_dim)

                if l == self.num_layers_b - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight, 0.0)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :ori_in], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, ori_in:], 0.0)
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :-(dims_in - ori_in)], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims_in - ori_in):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

                if weight_norm and l < self.num_layers_b - 2:
                    lin = nn.utils.weight_norm(lin)

                setattr(self, "lin"+str(i_b)+"_b_"+str(l), lin)

        # latent code
        for i_b in range(self.n_blocks):
            lin = nn.Linear(d_feature, d_feature)
            torch.nn.init.constant_(lin.bias, 0.0)
            torch.nn.init.constant_(lin.weight, 0.0)
            setattr(self, "lin"+str(i_b)+"_c", lin)

        self.activation = nn.Softplus(beta=100)


    def forward(self, deformation_code, input_pts, alpha_ratio=1.0):
        batch_size = input_pts.shape[0]
        x = input_pts
        for i_b in range(self.n_blocks):
            form = (i_b // 3) % 2
            mode = i_b % 3

            lin = getattr(self, "lin"+str(i_b)+"_c")
            deform_code_ib = lin(deformation_code) + deformation_code
            deform_code_ib = deform_code_ib.repeat(batch_size, 1)
            # part a
            if form == 0:
                # zyx
                if mode == 0:
                    x_focus = x[:, [2]]
                    x_other = x[:, [0,1]]
                elif mode == 1:
                    x_focus = x[:, [1]]
                    x_other = x[:, [0,2]]
                else:
                    x_focus = x[:, [0]]
                    x_other = x[:, [1,2]]
            else:
                # xyz
                if mode == 0:
                    x_focus = x[:, [0]]
                    x_other = x[:, [1,2]]
                elif mode == 1:
                    x_focus = x[:, [1]]
                    x_other = x[:, [0,2]]
                else:
                    x_focus = x[:, [2]]
                    x_other = x[:, [0,1]]
            x_ori = x_other # xy
            if self.embed_fn_1 is not None:
                # Anneal
                x_other = self.embed_fn_1(x_other, alpha_ratio)
            x_other = torch.cat([x_other, deform_code_ib], dim=-1)
            x = x_other
            for l in range(0, self.num_layers_a - 1):
                lin = getattr(self, "lin"+str(i_b)+"_a_"+str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_other], 1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_a - 2:
                    x = self.activation(x)
            if self.scale_included:
                scale_x = torch.exp(-x[:,[1]])
                x_focus = scale_x * (x_focus - x[:,[0]])
            else:
                x_focus = (x_focus - x[:,[0]])

            # part b
            x_focus_ori = x_focus # z'
            if self.embed_fn_2 is not None:
                # Anneal
                x_focus = self.embed_fn_2(x_focus, alpha_ratio)
            x_focus = torch.cat([x_focus, deform_code_ib], dim=-1)
            x = x_focus
            for l in range(0, self.num_layers_b - 1):
                lin = getattr(self, "lin"+str(i_b)+"_b_"+str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_focus], 1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_b - 2:
                    x = self.activation(x)

            # scale_2d = x[:, 4:6]
            rot_2d = euler2rot_2dinv(x[:, [0]])
            trans_2d = x[:,1:3]
            if self.scale_included:
                scale_xy = torch.exp(-x[:, 3:])
                scale_matrix = torch.diag_embed(scale_xy)
                x_other = torch.bmm(scale_matrix,torch.bmm(rot_2d, (x_ori - trans_2d)[...,None])).squeeze(-1)
            else:
                x_other = torch.bmm(rot_2d, (x_ori - trans_2d)[...,None]).squeeze(-1)
            if form == 0:
                if mode == 0:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_other[:,[0]], x_focus_ori, x_other[:,[1]]], dim=-1)
                else:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)
            else:
                if mode == 0:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_other[:,[0]], x_focus_ori, x_other[:,[1]]], dim=-1)
                else:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)

        return x


    def inverse(self, deformation_code, input_pts, alpha_ratio):
        batch_size = input_pts.shape[0]
        x = input_pts
        for i_b in range(self.n_blocks):
            i_b = self.n_blocks - 1 - i_b # inverse
            form = (i_b // 3) % 2
            mode = i_b % 3

            lin = getattr(self, "lin"+str(i_b)+"_c")
            deform_code_ib = lin(deformation_code) + deformation_code
            deform_code_ib = deform_code_ib.repeat(batch_size, 1)
            # part b
            if form == 0:
                # axis: z -> y -> x
                if mode == 0:
                    x_focus = x[:, [0,1]]
                    x_other = x[:, [2]]
                elif mode == 1:
                    x_focus = x[:, [0,2]]
                    x_other = x[:, [1]]
                else:
                    x_focus = x[:, [1,2]]
                    x_other = x[:, [0]]
            else:
                # axis: x -> y -> z
                if mode == 0:
                    x_focus = x[:, [1,2]]
                    x_other = x[:, [0]]
                elif mode == 1:
                    x_focus = x[:, [0,2]]
                    x_other = x[:, [1]]
                else:
                    x_focus = x[:, [0,1]]
                    x_other = x[:, [2]]
            x_ori = x_other # z'
            if self.embed_fn_2 is not None:
                # Anneal
                x_other = self.embed_fn_2(x_other, alpha_ratio)
            x_other = torch.cat([x_other, deform_code_ib], dim=-1)
            x = x_other
            for l in range(0, self.num_layers_b - 1):
                lin = getattr(self, "lin"+str(i_b)+"_b_"+str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_other], 1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_b - 2:
                    x = self.activation(x)

            rot_2d = euler2rot_2d(x[:, [0]])
            trans_2d = x[:, 1:]
            x_focus = torch.bmm(rot_2d, x_focus[...,None]).squeeze(-1) + trans_2d

            # part a
            x_focus_ori = x_focus # xy
            if self.embed_fn_1 is not None:
                # Anneal
                x_focus = self.embed_fn_1(x_focus, alpha_ratio)
            x_focus = torch.cat([x_focus, deform_code_ib], dim=-1)
            x = x_focus
            for l in range(0, self.num_layers_a - 1):
                lin = getattr(self, "lin"+str(i_b)+"_a_"+str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_focus], 1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_a - 2:
                    x = self.activation(x)

            x_other = x_ori + x
            if form == 0:
                if mode == 0:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_focus_ori[:,[0]], x_other, x_focus_ori[:,[1]]], dim=-1)
                else:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)
            else:
                if mode == 0:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_focus_ori[:,[0]], x_other, x_focus_ori[:,[1]]], dim=-1)
                else:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)

        return x

class HashGridLayer(nn.Module):
    
    def __init__(self, T=19, num_levels=16, n_features_per_level = 2,  proj_dims=256, min_res = 16, per_level_scale = 1.5, otype='HashGrid', use_tcnn="True"):
        super().__init__()
        self.in_dim = 2 
        self.out_dim = proj_dims
        if use_tcnn:
            import tinycudann as tcnn
            self.model = tcnn.NetworkWithInputEncoding(
                    n_input_dims=self.in_dim,
                    n_output_dims=self.out_dim,
                    encoding_config={
                        "otype": "HashGrid",
                        "n_levels": num_levels,
                        "n_features_per_level": n_features_per_level,
                        "log2_hashmap_size": T,
                        "base_resolution": min_res,
                        "per_level_scale": per_level_scale,
                    },
                    network_config= {
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": 128,
                        "n_hidden_layers": 2
                    }
                )


    @property
    def proj_dims(self):
        return self._proj_dims * 3

    def forward(self, x):
        return self.model(x)
    
def signed_volume_of_tetra(vertices):
    a = vertices[:, 0]
    b = vertices[:, 1]
    c = vertices[:, 2]
    d = vertices[:, 3]
    return torch.det(torch.stack([b-a, c-a, d-a], dim=1))
    
if __name__ == "__main__":
    # test DeformNetwork    
    import meshio
    mesh = meshio.read("/home/juyonggroup/kevin2000/project/gaussian-splatting/nerf_synthetic/lego/points3d.msh")
    tetrahedra = {}
    tetrahedra["vertices"] = np.array(mesh.points, dtype=np.float32)
    
    for cell in mesh.cells:
        if cell.type == "tetra":
            tetrahedra["cells"] = np.array(cell.data, dtype=np.int32)
            break
    
    d_feature = 64
    d_in = 3
    d_out_1 = 2 # t,s
    d_out_2 = 5 # (r,t,s) in 2d
    n_blocks = 6
    d_hidden = 256
    n_layers = 3
    skip_in = []
    multires = 5
    weight_norm = True

    deform_network = DeformNetwork(d_feature, True, n_blocks, d_hidden, n_layers, skip_in, multires, weight_norm).cuda()
    print(deform_network)
    
    # def init_weights(m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.normal_(m.weight, mean=0.0, std=0.1)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0.1)

    # deform_network.apply(init_weights)
    # random init network
    output = torch.empty(0, 3).cuda()
    deformation_code = torch.randn(1, d_feature).cuda()
    for i in range(tetrahedra["vertices"].shape[0] // 4096 + 1):
        if i < tetrahedra["vertices"].shape[0] // 4096:
            input_pts = torch.from_numpy(tetrahedra["vertices"]).cuda()[i*4096:(i+1)*4096]
        else:
            input_pts = torch.from_numpy(tetrahedra["vertices"]).cuda()[i*4096:]
        
        alpha_ratio = 0.5
        output = torch.cat([output,deform_network(deformation_code, input_pts, alpha_ratio)],dim=0)
  
    output_inv = deform_network.inverse(deformation_code, output, alpha_ratio)
    print(output_inv)
    assert torch.allclose(output, output_inv)
    print("Pass DeformNetwork test")