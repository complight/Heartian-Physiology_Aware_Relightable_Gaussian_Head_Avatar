import torch
import torch.nn as nn

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

class ProgressiveBandFrequency(nn.Module):
    def __init__(self, in_channels: int, n_frequencies=12, no_masking_step=5000):
        super().__init__()
        self.N_freqs = n_frequencies
        self.in_channels, self.n_input_dims = in_channels, in_channels
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2 ** torch.linspace(0, self.N_freqs - 1, self.N_freqs)
        self.n_output_dims = self.in_channels * (len(self.funcs) * self.N_freqs)
        self.n_masking_step = no_masking_step
        self.cur_step = nn.Parameter(torch.tensor(-1), requires_grad=False)
        self.update_step(0)

    def forward(self, x):
        out = []
        for freq, mask in zip(self.freq_bands, self.mask):
            for func in self.funcs:
                out += [func(freq * x) * mask]
        return torch.cat(out, -1)

    def update_step(self, global_step):
        if global_step > self.cur_step.item():
            if self.n_masking_step <= 0 or global_step is None or not self.training:
                self.mask = torch.ones(self.N_freqs, dtype=torch.float32, device=torch.device("cuda:0"))
            else:
                self.mask = (1.0 - torch.cos(torch.pi* (global_step / self.n_masking_step * self.N_freqs - torch.arange(0, self.N_freqs, device=torch.device("cuda:0"))).clamp(0, 1))) / 2.0
                # print(f"Update mask of Freq: {global_step}/{self.n_masking_step} {self.mask}")
            self.cur_step.data = torch.ones_like(self.cur_step) * global_step



# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import  Optional#, Sequence,Literal
from jaxtyping import Float, Int, Shaped
from torch import Tensor,nn
from abc import abstractmethod
import numpy as np

TCNN_EXISTS=False
class FieldComponent(nn.Module):
    """Field modules that can be combined to store and compute the fields.

    Args:
        in_dim: Input dimension to module.
        out_dim: Output dimension to module.
    """

    def __init__(self, in_dim: Optional[int] = None, out_dim: Optional[int] = None) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def build_nn_modules(self) -> None:
        """Function instantiates any torch.nn members within the module.
        If none exist, do nothing."""

    def set_in_dim(self, in_dim: int) -> None:
        """Sets input dimension of encoding

        Args:
            in_dim: input dimension
        """
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        self.in_dim = in_dim

    def get_out_dim(self) -> int:
        """Calculates output dimension of encoding."""
        if self.out_dim is None:
            raise ValueError("Output dimension has not been set")
        return self.out_dim

    @abstractmethod
    def forward(self, in_tensor: Shaped[Tensor, "*bs input_dim"]) -> Shaped[Tensor, "*bs output_dim"]:
        """
        Returns processed tensor

        Args:
            in_tensor: Input tensor to process
        """
        raise NotImplementedError


class Encoding(FieldComponent):
    """Encode an input tensor. Intended to be subclassed

    Args:
        in_dim: Input dimension of tensor
    """

    def __init__(self, in_dim: int) -> None:
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        super().__init__(in_dim=in_dim)

    @classmethod
    def get_tcnn_encoding_config(cls) -> dict:
        """Get the encoding configuration for tcnn if implemented"""
        raise NotImplementedError("Encoding does not have a TCNN implementation")

    @abstractmethod
    def forward(self, in_tensor: Shaped[Tensor, "*bs input_dim"]) -> Shaped[Tensor, "*bs output_dim"]:
        """Call forward and returns and processed tensor

        Args:
            in_tensor: the input tensor to process
        """
        raise NotImplementedError


class HashEncoding(Encoding):
    """Hash encoding

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        max_res: Resolution of largest feature grid.
        log2_hashmap_size: Size of hash map is 2^log2_hashmap_size.
        features_per_level: Number of features per level.
        hash_init_scale: Value to initialize hash grid.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
        interpolation: Interpolation override for tcnn hashgrid. Not supported for torch unless linear.
    """

    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        hash_init_scale: float = 0.001,
        implementation  = "torch",#:Literal["tcnn", "torch"]
        interpolation  = None,#:Optional[Literal["Nearest", "Linear", "Smoothstep"]]
    ) -> None:
        super().__init__(in_dim=3)
        self.num_levels = num_levels
        self.min_res = min_res
        self.features_per_level = features_per_level
        self.hash_init_scale = hash_init_scale
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size = 2**log2_hashmap_size

        levels = torch.arange(num_levels)
        self.growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1)) if num_levels > 1 else 1
        self.scalings = torch.floor(min_res * self.growth_factor**levels)

        self.hash_offset = levels * self.hash_table_size

        self.tcnn_encoding = None
        self.hash_table = torch.empty(0)
        if implementation == "torch":
            self.build_nn_modules()
        elif implementation == "tcnn" and not TCNN_EXISTS:
            module_name="HashEncoding"
            
            print(f"[bold yellow]WARNING: Using a slow implementation for the {module_name} module. ")
            print(
                "[bold yellow]:person_running: :person_running: "
                + "Install tcnn for speedups :person_running: :person_running:"
            )
            print("[yellow]pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch")
            
            self.build_nn_modules()
        elif implementation == "tcnn":
            encoding_config = self.get_tcnn_encoding_config(
                num_levels=self.num_levels,
                features_per_level=self.features_per_level,
                log2_hashmap_size=self.log2_hashmap_size,
                min_res=self.min_res,
                growth_factor=self.growth_factor,
                interpolation=interpolation,
            )
            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config,
            )

        if self.tcnn_encoding is None:
            assert (
                interpolation is None or interpolation == "Linear"
            ), f"interpolation '{interpolation}' is not supported for torch encoding backend"

    def build_nn_modules(self) -> None:
        """Initialize the torch version of the hash encoding."""
        self.hash_table = torch.rand(size=(self.hash_table_size * self.num_levels, self.features_per_level)) * 2 - 1
        self.hash_table *= self.hash_init_scale
        self.hash_table = nn.Parameter(self.hash_table)

    @classmethod
    def get_tcnn_encoding_config(
        cls, num_levels, features_per_level, log2_hashmap_size, min_res, growth_factor, interpolation=None
    ) -> dict:
        """Get the encoding configuration for tcnn if implemented"""
        encoding_config = {
            "otype": "HashGrid",
            "n_levels": num_levels,
            "n_features_per_level": features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": min_res,
            "per_level_scale": growth_factor,
        }
        if interpolation is not None:
            encoding_config["interpolation"] = interpolation
        return encoding_config

    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def hash_fn(self, in_tensor: Int[Tensor, "*bs num_levels 3"]) -> Shaped[Tensor, "*bs num_levels"]:
        """Returns hash tensor using method described in Instant-NGP

        Args:
            in_tensor: Tensor to be hashed
        """

        # min_val = torch.min(in_tensor)
        # max_val = torch.max(in_tensor)
        # assert min_val >= 0.0
        # assert max_val <= 1.0

        in_tensor = in_tensor * torch.tensor([1, 2654435761, 805459861]).to(in_tensor.device)
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.hash_table_size
        x += self.hash_offset.to(x.device)
        return x

    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""

        assert in_tensor.shape[-1] == 3
        in_tensor = in_tensor[..., None, :]  # [..., 1, 3]
        scaled = in_tensor * self.scalings.view(-1, 1).to(in_tensor.device)  # [..., L, 3]
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f

        hashed_0 = self.hash_fn(scaled_c)  # [..., num_levels]
        hashed_1 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))

        f_0 = self.hash_table[hashed_0]  # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = f0312 * offset[..., 2:3] + f4756 * (
            1 - offset[..., 2:3]
        )  # [..., num_levels, features_per_level]

        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)