import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from copy import deepcopy
from functools import partial
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.file_utils import ModelOutput
import time
from dust3r.utils.misc import (
    fill_default_args,
    freeze_all_params,
    is_symmetrized,
    interleave,
    transpose_to_landscape,
)
from dust3r.heads import head_factory
from dust3r.utils.camera import PoseEncoder
from dust3r.patch_embed import get_patch_embed
import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet, CrocoConfig  # noqa
from dust3r.blocks import (
    Block,
    DecoderBlock,
    Mlp,
    Attention,
    CrossAttention,
    DropPath,
)  # noqa

inf = float("inf")
from accelerate.logging import get_logger

from einops import rearrange
from dust3r.utils.device import to_cpu, to_gpu

printer = get_logger(__name__, log_level="DEBUG")


@dataclass
class ARCroco3DStereoOutput(ModelOutput):
    """
    Custom output class for ARCroco3DStereo.
    """

    ress: Optional[List[Any]] = None
    views: Optional[List[Any]] = None


def strip_module(state_dict):
    """
    Removes the 'module.' prefix from the keys of a state_dict.
    Args:
        state_dict (dict): The original state_dict with possible 'module.' prefixes.
    Returns:
        OrderedDict: A new state_dict with 'module.' prefixes removed.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def load_model(model_path, device, verbose=True):
    if verbose:
        print("... loading model from", model_path)
    ckpt = torch.load(model_path, map_location="cpu")
    args = ckpt["args"].model.replace(
        "ManyAR_PatchEmbed", "PatchEmbedDust3R"
    )  # ManyAR only for aspect ratio not consistent
    if "landscape_only" not in args:
        args = args[:-2] + ", landscape_only=False))"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt["model"], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class ARCroco3DStereoConfig(PretrainedConfig):
    model_type = "arcroco_3d_stereo"

    def __init__(
        self,
        output_mode="pts3d",
        head_type="linear",  # or dpt
        depth_mode=("exp", -float("inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        pose_mode=("exp", -float("inf"), float("inf")),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",
        ray_enc_depth=2,
        state_size=324,
        local_mem_size=256,
        state_pe="2d",
        state_dec_num_heads=16,
        depth_head=False,
        rgb_head=False,
        pose_conf_head=False,
        pose_head=False,
        model_update_type="cut3r",
        **croco_kwargs,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.freeze = freeze
        self.landscape_only = landscape_only
        self.patch_embed_cls = patch_embed_cls
        self.ray_enc_depth = ray_enc_depth
        self.state_size = state_size
        self.state_pe = state_pe
        self.state_dec_num_heads = state_dec_num_heads
        self.local_mem_size = local_mem_size
        self.depth_head = depth_head
        self.rgb_head = rgb_head
        self.pose_conf_head = pose_conf_head
        self.pose_head = pose_head
        self.model_update_type = model_update_type
        self.croco_kwargs = croco_kwargs


class LocalMemory(nn.Module):
    def __init__(
        self,
        size,
        k_dim,
        v_dim,
        num_heads,
        depth=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ) -> None:
        super().__init__()
        self.v_dim = v_dim
        self.proj_q = nn.Linear(k_dim, v_dim)
        self.masked_token = nn.Parameter(
            torch.randn(1, 1, v_dim) * 0.2, requires_grad=True
        ) # [1, 1, 768] pose mask token
        self.mem = nn.Parameter(
            torch.randn(1, size, 2 * v_dim) * 0.2, requires_grad=True
        ) # [1, 256, 1536] pose mem
        self.write_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.read_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

    def update_mem(self, mem, feat_k, feat_v, return_attn=False):
        """
        mem_k: [B, size, C]
        mem_v: [B, size, C]
        feat_k: [B, 1, C] global_img_feat
        feat_v: [B, 1, C] out_pose_feat
        """
        feat_k = self.proj_q(feat_k)  # [B, 1, C]
        feat = torch.cat([feat_k, feat_v], dim=-1)

        attention_maps = []
        for blk in self.write_blocks:
            mem, _, self_attn, cross_attn = blk(mem, feat, None, None, return_attn=return_attn)
            attention_maps.append((self_attn, cross_attn))
        return mem

    def inquire(self, query, mem, return_attn=False):
        x = self.proj_q(query)  # [B, 1, C]
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1) # [1, 1, 768 global_img_feat_i + 768 masked_token(pose)]
        attention_maps = []
        for blk in self.read_blocks:
            x, _, self_attn, cross_attn = blk(x, mem, None, None, return_attn=return_attn)
            attention_maps.append((self_attn, cross_attn))
        return x[..., -self.v_dim :]


class ARCroco3DStereo(CroCoNet):
    config_class = ARCroco3DStereoConfig
    base_model_prefix = "arcroco3dstereo"
    supports_gradient_checkpointing = True

    def __init__(self, config: ARCroco3DStereoConfig):
        self.gradient_checkpointing = False
        self.fixed_input_length = True
        config.croco_kwargs = fill_default_args(
            config.croco_kwargs, CrocoConfig.__init__
        )
        self.config = config
        self.patch_embed_cls = config.patch_embed_cls
        self.croco_args = config.croco_kwargs
        croco_cfg = CrocoConfig(**self.croco_args)
        super().__init__(croco_cfg)
        self.enc_blocks_ray_map = nn.ModuleList(
            [
                Block(
                    self.enc_embed_dim,
                    16,
                    4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    rope=self.rope,
                )
                for _ in range(config.ray_enc_depth)
            ]
        )
        self.enc_norm_ray_map = nn.LayerNorm(self.enc_embed_dim, eps=1e-6)
        self.dec_num_heads = self.croco_args["dec_num_heads"]
        self.pose_head_flag = config.pose_head
        if self.pose_head_flag:
            self.pose_token = nn.Parameter(
                torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True
            ) # [1, 1, 768]
            self.pose_retriever = LocalMemory(
                size=config.local_mem_size,
                k_dim=self.enc_embed_dim,
                v_dim=self.dec_embed_dim,
                num_heads=self.dec_num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                attn_drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                rope=None,
            )
        self.register_tokens = nn.Embedding(config.state_size, self.enc_embed_dim) # init state tokens [768, 1024]
        self.state_size = config.state_size
        self.state_pe = config.state_pe
        self.masked_img_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self.masked_ray_map_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self._set_state_decoder(
            self.enc_embed_dim,
            self.dec_embed_dim,
            config.state_dec_num_heads,
            self.dec_depth,
            self.croco_args.get("mlp_ratio", None),
            self.croco_args.get("norm_layer", None),
            self.croco_args.get("norm_im2_in_dec", None),
        )
        self.set_downstream_head(
            config.output_mode,
            config.head_type,
            config.landscape_only,
            config.depth_mode,
            config.conf_mode,
            config.pose_mode,
            config.depth_head,
            config.rgb_head,
            config.pose_conf_head,
            config.pose_head,
            **self.croco_args,
        )
        self.set_freeze(config.freeze)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            try:
                model = super(ARCroco3DStereo, cls).from_pretrained(
                    pretrained_model_name_or_path, **kw
                )
            except TypeError as e:
                raise Exception(
                    f"tried to load {pretrained_model_name_or_path} from huggingface, but failed"
                )
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )
        self.patch_embed_ray_map = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=6
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_state_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth_state = dec_depth
        self.dec_embed_dim_state = dec_embed_dim
        self.decoder_embed_state = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks_state = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm_state = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        if all(k.startswith("module") for k in ckpt):
            ckpt = strip_module(ckpt)
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks_state") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks_state")] = value
        try:
            return super().load_state_dict(new_ckpt, **kw)
        except:
            try:
                new_new_ckpt = {
                    k: v
                    for k, v in new_ckpt.items()
                    if not k.startswith("dec_blocks")
                    and not k.startswith("dec_norm")
                    and not k.startswith("decoder_embed")
                }
                return super().load_state_dict(new_new_ckpt, **kw)
            except:
                new_new_ckpt = {}
                for key in new_ckpt:
                    if key in self.state_dict():
                        if new_ckpt[key].size() == self.state_dict()[key].size():
                            new_new_ckpt[key] = new_ckpt[key]
                        else:
                            printer.info(
                                f"Skipping '{key}': size mismatch (ckpt: {new_ckpt[key].size()}, model: {self.state_dict()[key].size()})"
                            )
                    else:
                        printer.info(f"Skipping '{key}': not found in model")
                return super().load_state_dict(new_new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "mask": [self.mask_token] if hasattr(self, "mask_token") else [],
            "encoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
            ],
            "encoder_and_head": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.downstream_head,
            ],
            "encoder_and_decoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
                self.register_tokens,
                self.decoder_embed_state,
                self.decoder_embed,
                self.dec_norm,
                self.dec_norm_state,
            ],
            "decoder": [
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
            ],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        pose_mode,
        depth_head,
        rgb_head,
        pose_conf_head,
        pose_head,
        patch_size,
        img_size,
        **kw,
    ):
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.downstream_head = head_factory(
            head_type,
            output_mode,
            self,
            has_conf=bool(conf_mode),
            has_depth=bool(depth_head),
            has_rgb=bool(rgb_head),
            has_pose_conf=bool(pose_conf_head),
            has_pose=bool(pose_head),
        )
        self.head = transpose_to_landscape(
            self.downstream_head, activate=landscape_only
        )

    def _encode_image(self, image, true_shape):
        x, pos = self.patch_embed(image, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm(x)
        return [x], pos, None

    def _encode_ray_map(self, ray_map, true_shape):
        x, pos = self.patch_embed_ray_map(ray_map, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks_ray_map:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm_ray_map(x)
        return [x], pos, None

    def _encode_state(self, image_tokens, image_pos):
        batch_size = image_tokens.shape[0]
        state_feat = self.register_tokens(
            torch.arange(self.state_size, device=image_pos.device)
        ) # [768, 1024]
        if self.state_pe == "1d":
            state_pos = (
                torch.tensor(
                    [[i, i] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )  # .long()
        elif self.state_pe == "2d":
            width = int(self.state_size**0.5)
            width = width + 1 if width % 2 == 1 else width
            state_pos = (
                torch.tensor(
                    [[i // width, i % width] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        elif self.state_pe == "none":
            state_pos = None
        state_feat = state_feat[None].expand(batch_size, -1, -1)
        return state_feat, state_pos, None

    def _encode_views(self, views, img_mask=None, ray_mask=None):
        device = views[0]["img"].device
        batch_size = views[0]["img"].shape[0]
        given = True
        if img_mask is None and ray_mask is None:
            given = False
        if not given:
            img_mask = torch.stack(
                [view["img_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
            ray_mask = torch.stack(
                [view["ray_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
        imgs = torch.stack(
            [view["img"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, C, H, W)
        ray_maps = torch.stack(
            [view["ray_map"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, H, W, C)
        shapes = []
        for view in views:
            if "true_shape" in view:
                shapes.append(view["true_shape"])
            else:
                shape = torch.tensor(view["img"].shape[-2:], device=device)
                shapes.append(shape.unsqueeze(0).repeat(batch_size, 1))
        shapes = torch.stack(shapes, dim=0).to(
            imgs.device
        )  # Shape: (num_views, batch_size, 2)
        imgs = imgs.view(
            -1, *imgs.shape[2:]
        )  # Shape: (num_views * batch_size, C, H, W)
        ray_maps = ray_maps.view(
            -1, *ray_maps.shape[2:]
        )  # Shape: (num_views * batch_size, H, W, C)
        shapes = shapes.view(-1, 2)  # Shape: (num_views * batch_size, 2)
        img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
        ray_masks_flat = ray_mask.view(-1)
        selected_imgs = imgs[img_masks_flat]
        selected_shapes = shapes[img_masks_flat]
        if selected_imgs.size(0) > 0:
            img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
        else:
            raise NotImplementedError
        full_out = [
            torch.zeros(
                len(views) * batch_size, *img_out[0].shape[1:], device=img_out[0].device
            )
            for _ in range(len(img_out))
        ]
        full_pos = torch.zeros(
            len(views) * batch_size,
            *img_pos.shape[1:],
            device=img_pos.device,
            dtype=img_pos.dtype,
        )
        for i in range(len(img_out)):
            full_out[i][img_masks_flat] += img_out[i]
            full_out[i][~img_masks_flat] += self.masked_img_token
        full_pos[img_masks_flat] += img_pos
        ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
        selected_ray_maps = ray_maps[ray_masks_flat]
        selected_shapes_ray = shapes[ray_masks_flat]
        if selected_ray_maps.size(0) > 0:
            ray_out, ray_pos, _ = self._encode_ray_map(
                selected_ray_maps, selected_shapes_ray
            )
            assert len(ray_out) == len(full_out), f"{len(ray_out)}, {len(full_out)}"
            for i in range(len(ray_out)):
                full_out[i][ray_masks_flat] += ray_out[i]
                full_out[i][~ray_masks_flat] += self.masked_ray_map_token
            full_pos[ray_masks_flat] += (
                ray_pos * (~img_masks_flat[ray_masks_flat][:, None, None]).long()
            )
        else:
            raymaps = torch.zeros(
                1, 6, imgs[0].shape[-2], imgs[0].shape[-1], device=img_out[0].device
            )
            ray_mask_flat = torch.zeros_like(img_masks_flat)
            ray_mask_flat[:1] = True
            ray_out, ray_pos, _ = self._encode_ray_map(raymaps, shapes[ray_mask_flat])
            for i in range(len(ray_out)):
                full_out[i][ray_mask_flat] += ray_out[i] * 0.0
                full_out[i][~ray_mask_flat] += self.masked_ray_map_token * 0.0
        return (
            shapes.chunk(len(views), dim=0),
            [out.chunk(len(views), dim=0) for out in full_out],
            full_pos.chunk(len(views), dim=0),
        )

    def _decoder(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose, return_attn):
        final_output = [(f_state, f_img)]  # before projection
        assert f_state.shape[-1] == self.dec_embed_dim
        f_img = self.decoder_embed(f_img) # Linear: [1, 576, 1024] -> [1, 576, 768]
        if self.pose_head_flag:
            assert f_pose is not None and pos_pose is not None
            f_img = torch.cat([f_pose, f_img], dim=1) # [1, 1 + 576, 768]
            pos_img = torch.cat([pos_pose, pos_img], dim=1) # [1, 1 + 576, 2]
        final_output.append((f_state, f_img))
        attention_maps = []
        for blk_state, blk_img in zip(self.dec_blocks_state, self.dec_blocks):
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                f_state, _, self_attn_state, cross_attn_state = checkpoint(
                    blk_state,
                    *final_output[-1][::+1],
                    pos_state,
                    pos_img,
                    return_attn,
                    use_reentrant=not self.fixed_input_length,
                )
                f_img, _, self_attn_img, cross_attn_img = checkpoint(
                    blk_img,
                    *final_output[-1][::-1],
                    pos_img,
                    pos_state,
                    return_attn,
                    use_reentrant=not self.fixed_input_length,
                )
            else:
                f_state, _, self_attn_state, cross_attn_state = blk_state(*final_output[-1][::+1], pos_state, pos_img, return_attn=return_attn)
                f_img, _, self_attn_img, cross_attn_img = blk_img(*final_output[-1][::-1], pos_img, pos_state, return_attn=return_attn)
            final_output.append((f_state, f_img))
            attention_maps.append((self_attn_state, cross_attn_state, self_attn_img, cross_attn_img))
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = (
            self.dec_norm_state(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )
        return zip(*final_output), zip(*attention_maps)

    def _downstream_head(self, decout, img_shape, **kwargs):
        B, S, D = decout[-1].shape
        head = getattr(self, f"head")
        return head(decout, img_shape, **kwargs)

    def _init_state(self, image_tokens, image_pos):
        """
        Current Version: input the first frame img feature and pose to initialize the state feature and pose
        # [1, 768, 768] [1, 768, 2]
        """
        state_feat, state_pos, _ = self._encode_state(image_tokens, image_pos)
        state_feat = self.decoder_embed_state(state_feat) # Linear: [1, 768, 1024] -> [1, 768, 768]
        return state_feat, state_pos

    def _recurrent_rollout(
        self,
        state_feat,
        state_pos,
        current_feat,
        current_pos,
        pose_feat,
        pose_pos,
        init_state_feat,
        img_mask=None,
        reset_mask=None,
        update=None,
        return_attn=False,
    ):
        (new_state_feat, dec), (self_attn_state, cross_attn_state, self_attn_img, cross_attn_img) = self._decoder(
            state_feat, state_pos, current_feat, current_pos, pose_feat, pose_pos, return_attn
        )
        new_state_feat = new_state_feat[-1]
        return new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img

    def _get_img_level_feat(self, feat):
        return torch.mean(feat, dim=1, keepdim=True)

    # tbptt training encoder: Truncated Backpropagation Through Time
    def _forward_encoder(self, views):
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = self._init_state(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        return (feat, pos, shape), (
            init_state_feat,
            init_mem,
            state_feat,
            state_pos,
            mem,
        )

    # tbptt training decoder step: Truncated Backpropagation Through Time
    def _forward_decoder_step(
        self,
        views,
        i,
        feat_i,
        pos_i,
        shape_i,
        init_state_feat,
        init_mem,
        state_feat,
        state_pos,
        mem,
    ):
        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            if i == 0:
                pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            pose_feat_i = None
            pose_pos_i = None
        new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=views[i]["img_mask"],
            reset_mask=views[i]["reset"],
            update=views[i].get("update", None),
            return_attn=False,
        )
        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = self.pose_retriever.update_mem(
            mem, global_img_feat_i, out_pose_feat_i
        )
        head_input = [
            dec[0].float(),
            dec[self.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.dec_depth].float(),
        ]
        res = self._downstream_head(head_input, shape_i, pos=pos_i)
        img_mask = views[i]["img_mask"]
        update = views[i].get("update", None)
        if update is not None:
            update_mask = img_mask & update  # if don't update, then whatever img_mask
        else:
            update_mask = img_mask
        update_mask = update_mask[:, None, None].float()
        state_feat = new_state_feat * update_mask + state_feat * (
            1 - update_mask
        )  # update global state
        mem = new_mem * update_mask + mem * (1 - update_mask)  # then update local state
        reset_mask = views[i]["reset"]
        if reset_mask is not None:
            reset_mask = reset_mask[:, None, None].float()
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
            mem = init_mem * reset_mask + mem * (1 - reset_mask)
        return res, (state_feat, mem)

    # training and testing
    def _forward_impl(self, views, ret_state=False):
        # [B, C, H, W] -> [B, H/16*W/16, 1024]
        shape, feat_ls, pos = self._encode_views(views) # [15, 3, 288, 512] -> feat [15, 576, 1024], pos [15, 576, 2]
        feat = feat_ls[-1]
        state_feat, state_pos = self._init_state(feat[0], pos[0]) # init state feat [1, 768, 768], state_pos [1, 768, 2]
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1) # [1, 256, 1536] init pose mem
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]
        ress = []
        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]
            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i) # avg pool: [1, 576, 1024] -> [1, 1, 1024]
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1) # [1, 1, 768] init pose token
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem) 
                    # [1, 1, 768] use [global_img_feat_i, masked_token(pose)] as query, cross-attend mem, get pose_feat_i
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                ) # [1, 1, 2]
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
                state_feat, # [1, 768, 768]
                state_pos, # [1, 768, 2]
                feat_i, # [1, 576, 1024]
                pos_i, # [1, 576, 2]
                pose_feat_i, # [1, 1, 768] coarse pose token from pose_retriever
                pose_pos_i, # [1, 1, 2]
                init_state_feat,
                img_mask=views[i]["img_mask"],
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
                return_attn=True,
            ) # [1, 768, 768]
            out_pose_feat_i = dec[-1][:, 0:1] # [1, 1, 768] refined pose token from dust3r
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            ) # [1, 256, 1536] use mem as query, cross-attend [global_img_feat_i, out_pose_feat_i], get new_mem
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(), # [1, 576, 1024]
                dec[self.dec_depth * 2 // 4][:, 1:].float(), # [1, 576, 768]
                dec[self.dec_depth * 3 // 4][:, 1:].float(), # [1, 576, 768]
                dec[self.dec_depth].float(), # [1, 1 + 576, 768]
            ]
            res = self._downstream_head(head_input, shape[i], pos=pos_i)
            ress.append(res)
            img_mask = views[i]["img_mask"]
            update = views[i].get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()

            # update with learning rate
            update_type = self.config.model_update_type

            # Extract depth for geo gate types
            if update_type in ("cut3r_geogate", "ttt3r_geogate",
                               "cut3r_joint", "ttt3r_joint",
                               "ttt3r_brake_geo"):
                curr_depth = res['pts3d_in_self_view'][0, :, :, 2]  # [H, W]

            if i == 0:
                update_mask1 = update_mask
                # Initialize spectral state
                if update_type in ("ttt3r_spectral", "cut3r_spectral",
                                   "cut3r_joint", "ttt3r_joint"):
                    spectral_state = {
                        'ema': state_feat.clone(),
                        'running_energy': torch.zeros(
                            1, state_feat.shape[1], 1,
                            device=state_feat.device),
                    }
                # Initialize geo gate state
                if update_type in ("cut3r_geogate", "ttt3r_geogate",
                                   "cut3r_joint", "ttt3r_joint",
                                   "ttt3r_brake_geo"):
                    geo_state = {'prev_depth': curr_depth.detach().clone()}
                # Initialize L2 norm gate state
                if update_type == "ttt3r_l2gate":
                    l2_state = {
                        'running_energy': torch.zeros(
                            1, state_feat.shape[1], 1,
                            device=state_feat.device),
                    }
                # Initialize momentum gate state
                if update_type in ("ttt3r_momentum", "ttt3r_brake_geo"):
                    momentum_state = {}
                # Initialize ortho state
                if update_type == "ttt3r_ortho":
                    ortho_state = {}
            else:
                if update_type == "cut3r":
                    update_mask1 = update_mask
                elif update_type == "ttt3r":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)') # [12, 16, 768, 1 + 576] -> [1, 768, 1 + 576, 12*16]
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    update_mask1 = update_mask * torch.sigmoid(state_query_img_key)[..., None] * 1.0
                elif update_type == "ttt3r_conf":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    conf_scale = getattr(self.config, 'conf_gate_scale', 10.0)
                    if "conf_self" in res:
                        mean_conf = res["conf_self"].mean()
                    elif "conf" in res:
                        mean_conf = res["conf"].mean()
                    else:
                        mean_conf = torch.tensor(conf_scale, device=feat.device)
                    conf_gate = torch.clamp(mean_conf / conf_scale, 0.0, 1.0)
                    update_mask1 = update_mask * ttt3r_mask * conf_gate
                elif update_type == "ttt3r_l2gate":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    alpha = self._l2_norm_gate(
                        state_feat, new_state_feat, l2_state, self.config)
                    update_mask1 = update_mask * ttt3r_mask * alpha
                elif update_type == "ttt3r_random":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    random_p = getattr(self.config, 'random_gate_p', 0.5)
                    update_mask1 = update_mask * ttt3r_mask * random_p
                elif update_type == "ttt3r_momentum":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    m_gate = self._momentum_gate(
                        state_feat, new_state_feat, momentum_state, self.config)
                    update_mask1 = update_mask * ttt3r_mask * m_gate
                elif update_type == "ttt3r_brake_geo":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    m_gate = self._momentum_gate(
                        state_feat, new_state_feat, momentum_state, self.config)
                    g_geo = self._geo_consistency_gate(curr_depth, geo_state, self.config)
                    update_mask1 = update_mask * ttt3r_mask * m_gate * g_geo
                elif update_type == "cut3r_spectral":
                    alpha = self._spectral_modulation(
                        state_feat, new_state_feat, spectral_state, self.config)
                    update_mask1 = update_mask * alpha
                elif update_type == "ttt3r_spectral":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    alpha = self._spectral_modulation(
                        state_feat, new_state_feat, spectral_state, self.config)
                    update_mask1 = update_mask * ttt3r_mask * alpha
                elif update_type == "cut3r_geogate":
                    g_geo = self._geo_consistency_gate(curr_depth, geo_state, self.config)
                    update_mask1 = update_mask * g_geo
                elif update_type == "ttt3r_geogate":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    g_geo = self._geo_consistency_gate(curr_depth, geo_state, self.config)
                    update_mask1 = update_mask * ttt3r_mask * g_geo
                elif update_type == "cut3r_joint":
                    alpha = self._spectral_modulation(
                        state_feat, new_state_feat, spectral_state, self.config)
                    g_geo = self._geo_consistency_gate(curr_depth, geo_state, self.config)
                    update_mask1 = update_mask * alpha * g_geo
                elif update_type == "ttt3r_joint":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    alpha = self._spectral_modulation(
                        state_feat, new_state_feat, spectral_state, self.config)
                    g_geo = self._geo_consistency_gate(curr_depth, geo_state, self.config)
                    update_mask1 = update_mask * ttt3r_mask * alpha * g_geo
                elif update_type == "ttt3r_ortho":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    updated = self._delta_ortho_update(
                        state_feat, new_state_feat, ortho_state, self.config)
                    new_state_feat = updated
                    update_mask1 = update_mask * ttt3r_mask
                else:
                    raise ValueError(f"Invalid model type: {update_type}")

            update_mask2 = update_mask
            state_feat = new_state_feat * update_mask1 + state_feat * (
                1 - update_mask1
            )  # update global state
            mem = new_mem * update_mask2 + mem * (
                1 - update_mask2
            )  # then update local state
            reset_mask = views[i]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
                # Only reset gate states when scene actually resets
                if reset_mask.any():
                    if update_type in ("ttt3r_spectral", "cut3r_spectral",
                                       "cut3r_joint", "ttt3r_joint"):
                        spectral_state = {
                            'ema': state_feat.clone(),
                            'running_energy': torch.zeros_like(
                                spectral_state['running_energy']),
                        }
                    if update_type == "ttt3r_l2gate":
                        l2_state = {
                            'running_energy': torch.zeros_like(
                                l2_state['running_energy']),
                        }
                    if update_type in ("ttt3r_momentum", "ttt3r_brake_geo"):
                        momentum_state = {}
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )
        if ret_state:
            return ress, views, all_state_args
        return ress, views

    def forward(self, views, ret_state=False):
        if ret_state:
            ress, views, state_args = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views), state_args
        else:
            ress, views = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views)

    # testing: generate rgb xyz condition on raymap
    def inference_step(
        self, view, state_feat, state_pos, init_state_feat, mem, init_mem
    ):
        batch_size = view["img"].shape[0]
        raymaps = []
        shapes = []
        for j in range(batch_size):
            assert view["ray_mask"][j]
            raymap = view["ray_map"][[j]].permute(0, 3, 1, 2)
            raymaps.append(raymap)
            shapes.append(
                view.get(
                    "true_shape",
                    torch.tensor(view["ray_map"].shape[-2:])[None].repeat(
                        view["ray_map"].shape[0], 1
                    ),
                )[[j]]
            )

        raymaps = torch.cat(raymaps, dim=0)
        shape = torch.cat(shapes, dim=0).to(raymaps.device)
        feat_ls, pos, _ = self._encode_ray_map(raymaps, shapes) # [1, 6, 384, 512] -> feat [1, 768, 1024], pos [1, 768, 2]

        feat_i = feat_ls[-1]
        pos_i = pos
        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            pose_feat_i = None
            pose_pos_i = None
        new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=view["img_mask"],
            reset_mask=view["reset"],
            update=view.get("update", None),
            return_attn=False,
        )

        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = self.pose_retriever.update_mem(
            mem, global_img_feat_i, out_pose_feat_i
        )
        assert len(dec) == self.dec_depth + 1
        head_input = [
            dec[0].float(),
            dec[self.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.dec_depth].float(),
        ]
        res = self._downstream_head(head_input, shape, pos=pos_i)
        return res, view

    # recurrent testing
    def forward_recurrent(self, views, device, ret_state=False):
        ress = []
        all_state_args = []
        for i, view in enumerate(views):
            device = view["img"].device
            batch_size = view["img"].shape[0]
            img_mask = view["img_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            ray_mask = view["ray_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            imgs = view["img"].unsqueeze(0)  # Shape: (1, batch_size, C, H, W)
            ray_maps = view["ray_map"].unsqueeze(
                0
            )  # Shape: (num_views, batch_size, H, W, C)
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )  # Shape: (num_views, batch_size, 2)
            imgs = imgs.view(
                -1, *imgs.shape[2:]
            )  # Shape: (num_views * batch_size, C, H, W)
            ray_maps = ray_maps.view(
                -1, *ray_maps.shape[2:]
            )  # Shape: (num_views * batch_size, H, W, C)
            shapes = shapes.view(-1, 2).to(
                imgs.device
            )  # Shape: (num_views * batch_size, 2)
            img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
            ray_masks_flat = ray_mask.view(-1)
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None
            ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
            selected_ray_maps = ray_maps[ray_masks_flat]
            selected_shapes_ray = shapes[ray_masks_flat]
            if selected_ray_maps.size(0) > 0:
                ray_out, ray_pos, _ = self._encode_ray_map(
                    selected_ray_maps, selected_shapes_ray
                )
            else:
                ray_out, ray_pos = None, None

            shape = shapes
            if img_out is not None and ray_out is None:
                feat_i = img_out[-1]
                pos_i = img_pos
            elif img_out is None and ray_out is not None:
                feat_i = ray_out[-1]
                pos_i = ray_pos
            elif img_out is not None and ray_out is not None:
                feat_i = img_out[-1] + ray_out[-1]
                pos_i = img_pos
            else:
                raise NotImplementedError

            if i == 0:
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()
                all_state_args.append(
                    (state_feat, state_pos, init_state_feat, mem, init_mem)
                )

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update", None),
                return_attn=False,
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape, pos=pos_i)
            ress.append(res)
            img_mask = view["img_mask"]
            update = view.get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()
            state_feat = new_state_feat * update_mask + state_feat * (
                1 - update_mask
            )  # update global state
            mem = new_mem * update_mask + mem * (
                1 - update_mask
            )  # then update local state
            reset_mask = view["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )
        if ret_state:
            return ress, views, all_state_args
        return ress, views

    @staticmethod
    def _spectral_modulation(state_feat, new_state_feat, spectral_state, config):
        """
        Compute per-token spectral modulation factor α ∈ [0, 1].

        Maintains EMA of state trajectory (low-pass).  Within each frame,
        tokens are ranked by high-freq residual energy: high-energy tokens
        (unstable) get suppressed (α → 0), low-energy tokens (stable) pass
        through (α → 1).

        Args:
            state_feat:     [1, n_state, D]  current state BEFORE update
            new_state_feat: [1, n_state, D]  proposed new state
            spectral_state: dict  (mutated in-place)
            config:         model config with spectral hyperparams
        Returns:
            alpha: [1, n_state, 1]  modulation factor per token
        """
        mu = getattr(config, 'spectral_ema_momentum', 0.95)
        tau = getattr(config, 'spectral_temperature', 2.0)

        # Update EMA with current (pre-update) state
        ema = spectral_state['ema']
        ema = mu * ema + (1 - mu) * state_feat  # [1, n_state, D]
        spectral_state['ema'] = ema

        # High-freq residual of the *proposed* new state
        high_freq = new_state_feat - ema                  # [1, n_state, D]
        energy = high_freq.norm(dim=-1, keepdim=True)     # [1, n_state, 1]

        # Cross-token ranking: normalize energy within each frame
        # High energy (unstable) → high percentile → alpha ↓
        e_max = energy.max(dim=1, keepdim=True).values    # [1, 1, 1]
        percentile = energy / (e_max + 1e-6)              # [1, n_state, 1] ∈ [0, 1]
        alpha = torch.sigmoid(-tau * (percentile - 0.5))  # ∈ (0, 1)
        return alpha

    @staticmethod
    def _mem_spectral_gate(spectral_change, mem_spectral_state, config):
        """
        Compute a scalar memory-write gate g_mem ∈ (0, 1) based on frame-level
        spectral_change relative to its EMA baseline.

        High spectral_change (novel / keyframe) → g_mem → 1.0 (write memory).
        Low spectral_change (redundant frame)    → g_mem → 0.0 (skip memory write).

        Args:
            spectral_change:     float, current frame spectral_change score
            mem_spectral_state:  dict {'ema': float, 'warmed_up': bool}  (mutated in-place)
            config:              model config
        Returns:
            g_mem: float scalar ∈ (0, 1)
        """
        gamma = getattr(config, 'mem_gate_ema_gamma', 0.95)
        tau   = getattr(config, 'mem_gate_tau', 3.0)
        # Hard threshold ratio: frames below this fraction of the EMA are suppressed
        skip_ratio = getattr(config, 'mem_gate_skip_ratio', 0.5)

        ema = mem_spectral_state.get('ema', None)
        if ema is None or not mem_spectral_state.get('warmed_up', False):
            # Warm-start: first call — initialise EMA to current energy
            mem_spectral_state['ema'] = spectral_change
            mem_spectral_state['warmed_up'] = True
            return 1.0  # always write on first frame

        # Update EMA
        ema = gamma * ema + (1 - gamma) * spectral_change
        mem_spectral_state['ema'] = ema

        # Soft gate: sigmoid centred at skip_ratio * ema
        # ratio = spectral_change / (skip_ratio * ema + eps)
        # g_mem = sigmoid(tau * (ratio - 1))
        eps = 1e-6
        ratio = spectral_change / (skip_ratio * ema + eps)
        g_mem = torch.sigmoid(torch.tensor(tau * (ratio - 1.0))).item()
        return g_mem

    @staticmethod
    def _l2_norm_gate(state_feat, new_state_feat, l2_state, config):
        """
        Naive baseline: gate using L2 norm of state delta instead of
        frequency-domain decomposition.  Same running-mean + sigmoid
        structure as _spectral_modulation, but without EMA low-pass.

        Args:
            state_feat:     [1, n_state, D]
            new_state_feat: [1, n_state, D]
            l2_state:       dict  (mutated in-place)
            config:         model config
        Returns:
            alpha: [1, n_state, 1]
        """
        gamma = getattr(config, 'spectral_running_momentum', 0.95)
        tau = getattr(config, 'spectral_temperature', 2.0)

        delta = new_state_feat - state_feat               # [1, n_state, D]
        energy = delta.norm(dim=-1, keepdim=True)          # [1, n_state, 1]

        running_e = l2_state['running_energy']
        if not l2_state.get('warmed_up', False):
            running_e = energy.clone()
            l2_state['warmed_up'] = True
        else:
            running_e = gamma * running_e + (1 - gamma) * energy
        l2_state['running_energy'] = running_e

        ratio = energy / (running_e + 1e-6)
        alpha = torch.sigmoid(-tau * (ratio - 1.0))
        return alpha

    @staticmethod
    def _momentum_gate(state_feat, new_state_feat, momentum_state, config):
        """
        Momentum-inspired gate: use cosine similarity between consecutive
        state deltas as a per-token gate.  When consecutive updates are
        aligned (cos > 0), the gate opens (accelerate); when they conflict
        (cos < 0), the gate closes (brake).

        Analogous to SGD momentum: reinforce consistent update directions.

        Args:
            state_feat:      [1, n_state, D]
            new_state_feat:  [1, n_state, D]
            momentum_state:  dict (mutated in-place, holds prev_delta)
            config:          model config
        Returns:
            gate: [1, n_state, 1]  values in (0, 1)
        """
        tau = getattr(config, 'momentum_tau', 2.0)
        delta = new_state_feat - state_feat  # [1, n_state, D]

        prev_delta = momentum_state.get('prev_delta', None)
        if prev_delta is None:
            momentum_state['prev_delta'] = delta.detach().clone()
            # First frame: no prior delta, use neutral gate
            return torch.ones(1, delta.shape[1], 1, device=delta.device) * 0.5

        cosine = torch.nn.functional.cosine_similarity(
            delta, prev_delta, dim=-1
        ).unsqueeze(-1)  # [1, n_state, 1]

        momentum_state['prev_delta'] = delta.detach().clone()
        # Inverted: high alignment → brake (state converging, don't disturb)
        #           low alignment  → update (new geometric info)
        return torch.sigmoid(-tau * cosine)

    @staticmethod
    def _delta_clip_update(state_feat, new_state_feat, clip_state, config):
        """
        Delta clipping: suppress per-token updates whose norm exceeds
        tau × EMA(norm). Normal updates pass through; outlier tokens
        (large sudden deltas, e.g. during rotation) are clipped.

        state_t = state_{t-1} + alpha * clip(delta_t, threshold)
        """
        alpha = getattr(config, 'clip_alpha', 0.33)
        tau   = getattr(config, 'clip_tau',   2.0)
        beta  = getattr(config, 'clip_beta',  0.99)

        delta      = new_state_feat - state_feat
        delta_norm = delta.norm(dim=-1, keepdim=True)

        ema_norm = clip_state.get('ema_norm', None)
        if ema_norm is None:
            clip_state['ema_norm'] = delta_norm.detach().clone()
            return state_feat + alpha * delta

        ema_norm = beta * ema_norm + (1.0 - beta) * delta_norm.detach()
        clip_state['ema_norm'] = ema_norm

        threshold = tau * ema_norm
        scale = (threshold / delta_norm.clamp(min=1e-8)).clamp(max=1.0)
        return state_feat + alpha * (delta * scale)

    @staticmethod
    def _delta_ortho_update(state_feat, new_state_feat, ortho_state, config):
        """
        Delta Orthogonalization (Drift Subtraction):
        Decompose delta into systematic drift component + novel component.
        Strongly dampen drift, preserve novel information.

        drift_dir  = EMA of normalized deltas  (the repeated direction)
        novel      = delta - proj(delta, drift_dir)   (perpendicular)
        state_t    = state_{t-1} + alpha_novel * novel + alpha_drift * drift
        """
        alpha_novel = getattr(config, 'ortho_alpha_novel', 0.5)
        alpha_drift = getattr(config, 'ortho_alpha_drift', 0.05)
        beta        = getattr(config, 'ortho_beta',        0.95)

        # Length-aware warmup: no drift suppression for first T0 frames,
        # linearly ramp up over warmup window
        t0       = getattr(config, 'ortho_warmup_t0', 0)
        warmup_w = getattr(config, 'ortho_warmup_window', 0)

        step = ortho_state.get('step', 0)
        ortho_state['step'] = step + 1

        delta = new_state_feat - state_feat  # [B, T, D]
        # Normalize per token
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        delta_dir  = delta / delta_norm       # [B, T, D]

        drift_dir = ortho_state.get('drift_dir', None)
        if drift_dir is None:
            ortho_state['drift_dir'] = delta_dir.detach().clone()
            return state_feat + alpha_novel * delta

        # Update drift direction EMA
        drift_dir = beta * drift_dir + (1.0 - beta) * delta_dir.detach()
        drift_dir = drift_dir / drift_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        ortho_state['drift_dir'] = drift_dir

        # Decompose: project delta onto drift direction
        proj_scalar   = (delta * drift_dir).sum(dim=-1, keepdim=True)  # [B, T, 1]
        drift_comp    = proj_scalar * drift_dir                         # [B, T, D]
        novel_comp    = delta - drift_comp                              # [B, T, D]

        # Adaptive α_drift: scale with running drift energy
        adaptive_mode = getattr(config, 'ortho_adaptive', '')
        if adaptive_mode:
            # Per-token drift energy = cos²(delta, drift_dir)
            cos_sim = (delta_dir * drift_dir).sum(dim=-1, keepdim=True)  # [B, T, 1]
            drift_energy = cos_sim ** 2  # [B, T, 1]
            # Update running EMA of drift energy
            ema_drift_e = ortho_state.get('ema_drift_energy', None)
            if ema_drift_e is None:
                ema_drift_e = drift_energy.detach().clone()
            else:
                ema_drift_e = beta * ema_drift_e + (1.0 - beta) * drift_energy.detach()
            ortho_state['ema_drift_energy'] = ema_drift_e

            if adaptive_mode == 'linear':
                effective_alpha_drift = alpha_drift + (alpha_novel - alpha_drift) * ema_drift_e
            elif adaptive_mode == 'match':
                effective_alpha_drift = alpha_novel * ema_drift_e + alpha_drift * (1.0 - ema_drift_e)
            elif adaptive_mode == 'threshold':
                use_uniform = (ema_drift_e > 0.5).float()
                effective_alpha_drift = alpha_novel * use_uniform + alpha_drift * (1.0 - use_uniform)
            else:
                effective_alpha_drift = alpha_drift
        else:
            effective_alpha_drift = alpha_drift

        # Apply warmup: blend effective_alpha_drift toward alpha_novel during early frames
        if t0 > 0 and step < t0 + warmup_w:
            if step < t0:
                # Pure warmup phase: no drift suppression, uniform dampening
                effective_alpha_drift = alpha_novel
            else:
                # Ramp phase: linearly interpolate from alpha_novel to target
                ramp = (step - t0) / max(warmup_w, 1)
                effective_alpha_drift = alpha_novel + (effective_alpha_drift - alpha_novel) * ramp

        return state_feat + alpha_novel * novel_comp + effective_alpha_drift * drift_comp

    @staticmethod
    def _true_momentum_update(state_feat, new_state_feat, momentum_state, config):
        """
        True SGD-style momentum for state updates. Instead of gating (scalar),
        this smooths the update direction (vector) via EMA of deltas.

        m_t = β * m_{t-1} + (1-β) * δ_t
        state_t = state_{t-1} + α * m_t

        Returns the smoothed state directly (NOT a gate).

        Args:
            state_feat:      [1, n_state, D]  current state
            new_state_feat:  [1, n_state, D]  proposed new state
            momentum_state:  dict (holds 'ema_delta')
            config:          model config (momentum_beta, momentum_lr)
        Returns:
            updated_state:   [1, n_state, D]
        """
        beta = getattr(config, 'momentum_beta', 0.9)
        lr = getattr(config, 'momentum_lr', 0.33)

        delta = new_state_feat - state_feat  # [1, n_state, D]

        ema_delta = momentum_state.get('ema_delta', None)
        if ema_delta is None:
            momentum_state['ema_delta'] = delta.detach().clone()
            # First frame: apply delta with lr directly
            return state_feat + lr * delta

        # Accumulate momentum
        ema_delta = beta * ema_delta + (1 - beta) * delta
        momentum_state['ema_delta'] = ema_delta.detach().clone()

        # Apply smoothed update
        return state_feat + lr * ema_delta

    @staticmethod
    def _centered_sharp_gate(state_feat, new_state_feat, momentum_state, config):
        """
        Centered sharp gate: separate overall dampening rate from per-token
        selectivity. Centers cosine at its mean so sigmoid operates near 0
        (max sensitivity), then scales by base_rate.

        gate = base_rate * sigmoid(-tau_sharp * (cos - mean(cos)))

        Args/Returns: same as _momentum_gate
        """
        base_rate = getattr(config, 'gate_base_rate', 0.33)
        tau_sharp = getattr(config, 'gate_tau_sharp', 5.0)

        delta = new_state_feat - state_feat  # [1, n_state, D]
        prev_delta = momentum_state.get('prev_delta', None)
        if prev_delta is None:
            momentum_state['prev_delta'] = delta.detach().clone()
            return torch.ones(1, delta.shape[1], 1, device=delta.device) * base_rate

        cosine = torch.nn.functional.cosine_similarity(
            delta, prev_delta, dim=-1
        )  # [1, n_state]

        cos_centered = cosine - cosine.mean(dim=-1, keepdim=True)  # zero-mean
        # sigmoid at 0 has max gradient → preserves token differences
        selectivity = torch.sigmoid(-tau_sharp * cos_centered)  # [1, n_state]

        momentum_state['prev_delta'] = delta.detach().clone()
        return (base_rate * selectivity).unsqueeze(-1)  # [1, n_state, 1]

    @staticmethod
    def _feature_novelty_gate(feat_i, novelty_state, config):
        """
        Feature novelty gate: use cosine similarity between consecutive
        encoder features (per-token) to detect input change.

        Tokens where new frame brings novel features → higher gate (update more).
        Tokens with similar features → lower gate (redundant, dampen).

        gate = base_rate * (1 - feat_sim) rescaled to [low, high]

        Args:
            feat_i:          [1, n_patches, D] — current frame encoder output
            novelty_state:   dict (holds prev_feat)
            config:          model config
        Returns:
            gate: [1, n_state, 1]  (n_state != n_patches typically, so we
                  interpolate or use mean pooling)
        """
        base_rate = getattr(config, 'novelty_base_rate', 0.33)
        tau_novelty = getattr(config, 'novelty_tau', 5.0)

        prev_feat = novelty_state.get('prev_feat', None)
        if prev_feat is None:
            novelty_state['prev_feat'] = feat_i.detach().clone()
            return None  # signal: use base_rate as scalar

        # Per-token cosine similarity between consecutive encoder outputs
        feat_sim = torch.nn.functional.cosine_similarity(
            feat_i, prev_feat, dim=-1
        )  # [1, n_patches]

        # Center and apply sharp sigmoid: novel patches → high gate
        sim_centered = feat_sim - feat_sim.mean(dim=-1, keepdim=True)
        # Positive sim_centered = more similar than average → dampen more
        novelty_gate = torch.sigmoid(-tau_novelty * sim_centered)  # [1, n_patches]

        novelty_state['prev_feat'] = feat_i.detach().clone()
        return (base_rate * novelty_gate).unsqueeze(-1)  # [1, n_patches, 1]

    @staticmethod
    def _geo_consistency_gate(curr_depth, geo_state, config):
        """
        Compute a scalar geometric consistency gate g_geo ∈ (0, 1) based on
        the low-frequency energy of the log-depth difference between
        consecutive frames (frequency-domain geometric consistency).

        Mirrors compute_frame_spectral_change but operates on predicted depth
        maps instead of RGB, unifying all three layers under frequency-domain
        analysis:
          Layer 1: LFE(RGB diff) → frame filtering
          Layer 2: token trajectory HF energy → state modulation
          Layer 3: LFE(depth diff) → state update gating  (this method)

        Stable depth structure → low LFE → g_geo → 1.0 (allow update).
        Sudden geometric change → high LFE → g_geo → 0.0 (suppress update).

        Args:
            curr_depth:  [H, W] tensor, predicted depth of current frame
            geo_state:   dict {'prev_depth': Tensor, 'ema': float}
                         (mutated in-place)
            config:      model config
        Returns:
            g_geo: float scalar ∈ (0, 1)
        """
        gamma = getattr(config, 'geo_gate_ema_gamma', 0.95)
        tau   = getattr(config, 'geo_gate_tau', 3.0)

        prev_depth = geo_state.get('prev_depth', None)
        geo_state['prev_depth'] = curr_depth.detach().clone()

        if prev_depth is None:
            return 1.0

        # Log-depth difference (scale-invariant)
        eps = 1e-4
        valid = (prev_depth > eps) & (curr_depth > eps)
        if valid.sum() < 100:
            return 1.0

        # Build full-size log-depth diff map (zero where invalid)
        log_diff = torch.zeros_like(curr_depth)
        log_diff[valid] = torch.log(curr_depth[valid]) - torch.log(prev_depth[valid])

        # Frequency-domain: low+mid frequency energy of depth diff
        # cutoff_ratio controls how much of the spectrum to include:
        #   1/8 = only low-freq (structural), 1/4 = low+mid (structural + geometric detail)
        cutoff_ratio = getattr(config, 'geo_gate_freq_cutoff', 4)  # denominator: H//4, W//4
        F = torch.fft.fft2(log_diff)
        power = F.abs() ** 2
        H, W = power.shape
        h_cut = max(1, H // cutoff_ratio)
        w_cut = max(1, W // cutoff_ratio)
        low_freq_energy = (power[:h_cut, :w_cut].sum() +
                           power[:h_cut, -w_cut:].sum() +
                           power[-h_cut:, :w_cut].sum() +
                           power[-h_cut:, -w_cut:].sum())
        change = low_freq_energy.item()

        # EMA baseline (warm-start)
        ema = geo_state.get('ema', None)
        if ema is None:
            geo_state['ema'] = change
            return 1.0

        ema = gamma * ema + (1 - gamma) * change
        geo_state['ema'] = ema

        # Gate: suppress when change >> baseline (geometric inconsistency)
        ratio = change / (ema + 1e-6)
        g_geo = torch.sigmoid(torch.tensor(-tau * (ratio - 1.0))).item()
        return g_geo

    @staticmethod
    def compute_frame_spectral_change(img_prev, img_curr):
        """
        Compute the low-frequency structural energy of the inter-frame difference.

        Returns the absolute low-frequency energy (not a ratio), so that
        redundant frames (tiny changes) produce small values and high-change frames
        (large structural changes) produce large values.  The caller is
        responsible for adaptive thresholding via a running mean.

        Args:
            img_prev: [B, C, H, W] float tensor, previous frame (values in [-1,1])
            img_curr: [B, C, H, W] float tensor, current frame
        Returns:
            low_freq_energy: scalar float ≥ 0 (un-normalised)
        """
        diff = img_curr - img_prev                        # [B, C, H, W]
        diff_mean = diff.mean(dim=(0, 1))                 # [H, W]

        F = torch.fft.fft2(diff_mean)
        power = F.abs() ** 2                              # [H, W]

        H, W = power.shape
        h_cut = max(1, H // 8)   # top 12.5% of spatial frequencies = low-freq
        w_cut = max(1, W // 8)

        low_freq_energy = (power[:h_cut, :w_cut].sum() +
                           power[:h_cut, -w_cut:].sum() +
                           power[-h_cut:, :w_cut].sum() +
                           power[-h_cut:, -w_cut:].sum())
        return low_freq_energy.item()

    @staticmethod
    def filter_views_by_spectral_change(views, skip_ratio=0.3, warmup=10,
                                always_keep_first=True, device='cpu'):
        """
        Adaptively filter a view sequence, skipping the least high-change frames.

        Uses a running mean of low-frequency structural energy as the reference.
        A frame is skipped if its energy falls below (skip_ratio * running_mean),
        i.e., it brings less than skip_ratio of the average structural change.
        A warmup period ensures the running mean is stable before filtering.

        Args:
            views:             list of view dicts (each has 'img' key [B,C,H,W])
            skip_ratio:        frames with energy < skip_ratio * running_mean → skip
            warmup:            number of initial frames always kept (to warm up stats)
            always_keep_first: always include views[0]
            device:            device for FFT computation
        Returns:
            kept_views:    filtered list of view dicts
            kept_indices:  original indices of kept frames
            novelties:     list of per-frame raw spectral_change energies (len = len(views))
        """
        kept_views = []
        kept_indices = []
        novelties = [0.0]   # frame 0: no previous frame

        running_mean = None
        gamma = 0.95        # EMA decay for running mean

        img_prev = None
        for i, view in enumerate(views):
            img = view['img'].float().to(device)
            if i == 0:
                img_prev = img
                if always_keep_first:
                    kept_views.append(view)
                    kept_indices.append(i)
                continue

            energy = ARCroco3DStereo.compute_frame_spectral_change(img_prev, img)
            novelties.append(energy)

            # Warm-start running mean
            if running_mean is None:
                running_mean = energy
            else:
                running_mean = gamma * running_mean + (1 - gamma) * energy

            # Always keep during warmup; afterwards skip low-spectral-change frames
            is_informative = (i < warmup) or (energy >= skip_ratio * running_mean)

            if is_informative:
                kept_views.append(view)
                kept_indices.append(i)
                img_prev = img  # advance reference only on kept frames

        return kept_views, kept_indices, novelties

    def forward_recurrent_lighter(self, views, device='cuda', ret_state=False):
        ress = []
        all_state_args = []
        reset_mask = False
        spectral_state = None      # initialized at frame 0
        mem_spectral_state = {}    # for B2 memory gate
        prev_img = None            # for B2 spectral_change computation
        geo_state = {}             # for B3 geometric consistency gate
        for i, _view in enumerate(views):
            view = to_gpu(_view, device)
            device = view["img"].device
            batch_size = view["img"].shape[0]
            img_mask = view["img_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            ray_mask = view["ray_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            imgs = view["img"].unsqueeze(0)  # Shape: (1, batch_size, C, H, W)
            ray_maps = view["ray_map"].unsqueeze(
                0
            )  # Shape: (num_views, batch_size, H, W, C)
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )  # Shape: (num_views, batch_size, 2)
            imgs = imgs.view(
                -1, *imgs.shape[2:]
            )  # Shape: (num_views * batch_size, C, H, W)
            ray_maps = ray_maps.view(
                -1, *ray_maps.shape[2:]
            )  # Shape: (num_views * batch_size, H, W, C)
            shapes = shapes.view(-1, 2).to(
                imgs.device
            )  # Shape: (num_views * batch_size, 2)
            img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
            ray_masks_flat = ray_mask.view(-1)
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None
            ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
            selected_ray_maps = ray_maps[ray_masks_flat]
            selected_shapes_ray = shapes[ray_masks_flat]
            if selected_ray_maps.size(0) > 0:
                ray_out, ray_pos, _ = self._encode_ray_map(
                    selected_ray_maps, selected_shapes_ray
                )
            else:
                ray_out, ray_pos = None, None

            shape = shapes
            if img_out is not None and ray_out is None:
                feat_i = img_out[-1]
                pos_i = img_pos
            elif img_out is None and ray_out is not None:
                feat_i = ray_out[-1]
                pos_i = ray_pos
            elif img_out is not None and ray_out is not None:
                feat_i = img_out[-1] + ray_out[-1]
                pos_i = img_pos
            else:
                raise NotImplementedError

            if i == 0:
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0 or reset_mask:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update", None),
                return_attn=True,
            )
            out_pose_feat_i = dec[-1][:, 0:1]

            # update mem
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )

            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape, pos=pos_i)
            res_cpu = to_cpu(res)
            ress.append(res_cpu)
            img_mask = view["img_mask"]
            update = view.get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()

            # update with learning rate
            update_type = self.config.model_update_type

            # B3: extract depth for geometric consistency gate
            curr_depth = res['pts3d_in_self_view'][0, :, :, 2]  # [H, W], still on GPU

            # B2: compute frame-level spectral_change for memory gate
            curr_img = view["img"].float()
            if i == 0 or reset_mask:
                update_mask1 = update_mask
                # Initialize spectral state at frame 0
                if update_type in ("ttt3r_spectral", "cut3r_spectral",
                                   "cut3r_joint", "ttt3r_joint"):
                    spectral_state = {
                        'ema': state_feat.clone(),
                        'running_energy': torch.zeros(
                            1, state_feat.shape[1], 1,
                            device=state_feat.device),
                    }
                # Reset mem gate state on scene reset
                if update_type in ("cut3r_memgate", "ttt3r_memgate"):
                    mem_spectral_state = {}
                # Reset geo gate state on scene reset
                if update_type in ("cut3r_geogate", "ttt3r_geogate",
                                   "cut3r_joint", "ttt3r_joint",
                                   "ttt3r_brake_geo"):
                    geo_state = {'prev_depth': curr_depth.detach().clone()}
                # Initialize L2 norm gate state
                if update_type == "ttt3r_l2gate":
                    l2_state = {
                        'running_energy': torch.zeros(
                            1, state_feat.shape[1], 1,
                            device=state_feat.device),
                    }
                # Initialize momentum gate state
                if update_type in ("ttt3r_momentum", "ttt3r_brake_geo",
                                   "ttt3r_centered", "ttt3r_true_momentum"):
                    momentum_state = {}
                # Initialize delta clip state
                if update_type == "ttt3r_delta_clip":
                    clip_state = {}
                # Initialize accumulated attention importance state
                if update_type == "ttt3r_attn_protect":
                    attn_protect_state = {}
                # Initialize feature novelty gate state
                if update_type == "ttt3r_novelty":
                    novelty_state = {}
                # Initialize memory novelty gate state
                if update_type == "ttt3r_mem_novelty":
                    mem_novelty_state = {}
                # Initialize delta orthogonalization state
                if update_type == "ttt3r_ortho":
                    ortho_state = {}
                prev_img = curr_img
            else:
                if update_type == "cut3r":
                    update_mask1 = update_mask
                elif update_type == "cut3r_taum_log":
                    # cut3r update + compute & log what TAUM gate would produce
                    update_mask1 = update_mask
                    if hasattr(self, '_taum_prev_new_state') and self._taum_prev_new_state is not None:
                        sc = (new_state_feat - self._taum_prev_new_state).norm(dim=-1).squeeze(0)
                        sc_norm = sc / sc.mean()
                        t_mask = torch.sigmoid(sc_norm - 1.5)
                        fi_n = feat_i / feat_i.norm(dim=-1, keepdim=True)
                        pf_n = self._taum_prev_feat / self._taum_prev_feat.norm(dim=-1, keepdim=True)
                        fd = 1.0 - (fi_n * pf_n).sum(dim=-1)
                        ca = rearrange(torch.cat(cross_attn_state, dim=0), 'l h ns ni -> 1 ns ni (l h)')[:,:,1:,:]
                        am = ca.mean(dim=-1).abs()
                        ss = (am * fd.unsqueeze(1)).max(dim=-1)[0].squeeze(0)
                        s_mask = torch.sigmoid(ss)
                        f_mask = t_mask * s_mask
                        if not hasattr(self, '_taum_log'):
                            self._taum_log = []
                        self._taum_log.append({
                            "frame": i, "temporal_mean": float(t_mask.mean()),
                            "temporal_std": float(t_mask.std()),
                            "spatial_mean": float(s_mask.mean()),
                            "spatial_std": float(s_mask.std()),
                            "final_mean": float(f_mask.mean()),
                            "final_std": float(f_mask.std()),
                            "sc_cv": float((sc.std() / sc.mean()).item()),
                        })
                    self._taum_prev_new_state = new_state_feat.clone().detach()
                    self._taum_prev_feat = feat_i.clone().detach()
                elif update_type == "ttt3r":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)') # [12, 16, 768, 1 + 576] -> [1, 768, 1 + 576, 12*16]
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    update_mask1 = update_mask * torch.sigmoid(state_query_img_key)[..., None] * 1.0
                elif update_type == "ttt3r_conf":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    conf_scale = getattr(self.config, 'conf_gate_scale', 10.0)
                    if "conf_self" in res:
                        mean_conf = res["conf_self"].mean()
                    elif "conf" in res:
                        mean_conf = res["conf"].mean()
                    else:
                        mean_conf = torch.tensor(conf_scale, device=device)
                    conf_gate = torch.clamp(mean_conf / conf_scale, 0.0, 1.0)
                    update_mask1 = update_mask * ttt3r_mask * conf_gate
                elif update_type == "ttt3r_l2gate":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    alpha = self._l2_norm_gate(
                        state_feat, new_state_feat, l2_state, self.config)
                    update_mask1 = update_mask * ttt3r_mask * alpha
                elif update_type == "ttt3r_random":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    random_p = getattr(self.config, 'random_gate_p', 0.5)
                    update_mask1 = update_mask * ttt3r_mask * random_p
                elif update_type == "ttt3r_momentum":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    m_gate = self._momentum_gate(
                        state_feat, new_state_feat, momentum_state, self.config)
                    update_mask1 = update_mask * ttt3r_mask * m_gate
                elif update_type == "ttt3r_brake_geo":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    m_gate = self._momentum_gate(
                        state_feat, new_state_feat, momentum_state, self.config)
                    g_geo = self._geo_consistency_gate(curr_depth, geo_state, self.config)
                    update_mask1 = update_mask * ttt3r_mask * m_gate * g_geo
                elif update_type == "ttt3r_true_momentum":
                    updated = self._true_momentum_update(
                        state_feat, new_state_feat, momentum_state, self.config)
                    new_state_feat = updated
                    update_mask1 = update_mask
                elif update_type == "ttt3r_delta_clip":
                    updated = self._delta_clip_update(
                        state_feat, new_state_feat, clip_state, self.config)
                    new_state_feat = updated
                    update_mask1 = update_mask
                elif update_type == "ttt3r_attn_protect":
                    # Accumulated attention protection (EWC-like):
                    # state tokens with high historical attention usage → protect from update
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    # Per-token attention magnitude (sum over image positions and heads)
                    attn_weight = cross_attn_state.abs().mean(dim=(-1, -2))  # [1, n_state]
                    beta_attn = getattr(self.config, 'attn_protect_beta', 0.95)
                    base_rate = getattr(self.config, 'attn_protect_base', 0.33)
                    imp = attn_protect_state.get('importance', None)
                    if imp is None:
                        attn_protect_state['importance'] = attn_weight.detach().clone()
                        imp = attn_weight.detach()
                    else:
                        imp = beta_attn * imp + (1 - beta_attn) * attn_weight.detach()
                        attn_protect_state['importance'] = imp
                    # Normalize importance to [0,1], invert: high imp → low gate
                    imp_min = imp.min(dim=-1, keepdim=True)[0]
                    imp_max = imp.max(dim=-1, keepdim=True)[0]
                    imp_norm = (imp - imp_min) / (imp_max - imp_min + 1e-8)
                    # gate: frequently-used tokens get low update; unused tokens get full update
                    gate = base_rate * (1.0 - imp_norm).unsqueeze(-1)  # [1, n_state, 1]
                    update_mask1 = update_mask * gate
                elif update_type == "ttt3r_centered":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    c_gate = self._centered_sharp_gate(
                        state_feat, new_state_feat, momentum_state, self.config)
                    update_mask1 = update_mask * ttt3r_mask * c_gate
                elif update_type == "ttt3r_novelty":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    n_gate = self._feature_novelty_gate(
                        feat_i, novelty_state, self.config)
                    if n_gate is None:
                        base_rate = getattr(self.config, 'novelty_base_rate', 0.33)
                        update_mask1 = update_mask * ttt3r_mask * base_rate
                    else:
                        # n_gate: [1, n_patches, 1], need to match state dim
                        if n_gate.shape[1] != state_feat.shape[1]:
                            n_gate_mean = n_gate.mean(dim=1, keepdim=True)
                            update_mask1 = update_mask * ttt3r_mask * n_gate_mean
                        else:
                            update_mask1 = update_mask * ttt3r_mask * n_gate
                elif update_type == "cut3r_spectral":
                    alpha = self._spectral_modulation(
                        state_feat, new_state_feat, spectral_state, self.config)
                    update_mask1 = update_mask * alpha
                elif update_type == "ttt3r_spectral":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    alpha = self._spectral_modulation(
                        state_feat, new_state_feat, spectral_state, self.config)
                    update_mask1 = update_mask * ttt3r_mask * alpha
                elif update_type == "cut3r_memgate":
                    update_mask1 = update_mask  # state update: same as cut3r
                elif update_type == "ttt3r_memgate":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    update_mask1 = update_mask * torch.sigmoid(state_query_img_key)[..., None] * 1.0
                elif update_type == "cut3r_geogate":
                    g_geo = self._geo_consistency_gate(curr_depth, geo_state, self.config)
                    update_mask1 = update_mask * g_geo
                elif update_type == "ttt3r_geogate":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    g_geo = self._geo_consistency_gate(curr_depth, geo_state, self.config)
                    update_mask1 = update_mask * ttt3r_mask * g_geo
                elif update_type == "cut3r_joint":
                    # Layer 2 (SIASU) × Layer 3 (GeoGate)
                    alpha = self._spectral_modulation(
                        state_feat, new_state_feat, spectral_state, self.config)
                    g_geo = self._geo_consistency_gate(curr_depth, geo_state, self.config)
                    update_mask1 = update_mask * alpha * g_geo
                elif update_type == "ttt3r_joint":
                    # TTT3R × Layer 2 (SIASU) × Layer 3 (GeoGate)
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    alpha = self._spectral_modulation(
                        state_feat, new_state_feat, spectral_state, self.config)
                    g_geo = self._geo_consistency_gate(curr_depth, geo_state, self.config)
                    update_mask1 = update_mask * ttt3r_mask * alpha * g_geo
                    # Optional gate logging for S4 visualization
                    if hasattr(self, '_gate_log') and self._gate_log is not None:
                        g_geo_cpu = g_geo.detach().cpu() if isinstance(g_geo, torch.Tensor) else torch.tensor(g_geo)
                        self._gate_log.append({
                            'frame': i,
                            'ttt3r_mask': ttt3r_mask.detach().cpu(),    # [1, 768, 1]
                            'alpha': alpha.detach().cpu(),               # [1, 768, 1]
                            'g_geo': g_geo_cpu,                          # scalar or [1]
                            'effective': update_mask1.detach().cpu(),     # [1, 768, 1]
                        })
                elif update_type == "ttt3r_mem_novelty":
                    # Memory Novelty Gate: gate state update based on how novel
                    # the current frame is relative to past frames, measured in
                    # the pose_retriever's projected feature space.
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    base_rate = getattr(self.config, 'mem_novelty_base', 0.33)
                    tau = getattr(self.config, 'mem_novelty_tau', 5.0)
                    beta = getattr(self.config, 'mem_novelty_beta', 0.95)
                    # Project current frame to memory key space [B, 1, v_dim]
                    q = self.pose_retriever.proj_q(global_img_feat_i)
                    ema_q = mem_novelty_state.get('ema_q', None)
                    if ema_q is None:
                        mem_novelty_state['ema_q'] = q.detach().clone()
                        update_mask1 = update_mask * ttt3r_mask * base_rate
                    else:
                        # Cosine similarity between current frame and running mean of past frames
                        sim = torch.nn.functional.cosine_similarity(
                            q.squeeze(1), ema_q.squeeze(1), dim=-1).mean()  # scalar
                        novelty = (1.0 - sim).clamp(0.0, 1.0)
                        # novelty→0 (familiar) → alpha→base_rate (small update)
                        # novelty→1 (novel)    → alpha→0.5 (full update)
                        alpha = base_rate + (0.5 - base_rate) * torch.sigmoid(tau * (novelty - 0.5))
                        update_mask1 = update_mask * ttt3r_mask * alpha
                        mem_novelty_state['ema_q'] = beta * ema_q + (1.0 - beta) * q.detach()
                elif update_type == "ttt3r_ortho":
                    # Delta Orthogonalization: decompose update into
                    # systematic drift (suppress) + novel direction (preserve)
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)')
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    updated = self._delta_ortho_update(
                        state_feat, new_state_feat, ortho_state, self.config)
                    new_state_feat = updated
                    update_mask1 = update_mask * ttt3r_mask
                else:
                    raise ValueError(f"Invalid model type: {update_type}")

            # B2: memory gate (applied for *_memgate types)
            if update_type in ("cut3r_memgate", "ttt3r_memgate") and i > 0 and not reset_mask:
                sc = self.compute_frame_spectral_change(prev_img, curr_img)
                g_mem = self._mem_spectral_gate(sc, mem_spectral_state, self.config)
                update_mask2 = update_mask * g_mem
            else:
                update_mask2 = update_mask
            prev_img = curr_img

            state_feat = new_state_feat * update_mask1 + state_feat * (
                1 - update_mask1
            )  # update global state
            mem = new_mem * update_mask2 + mem * (
                1 - update_mask2
            )  # then update local state (B2: gated by spectral_change for *_memgate types)

            reset_mask = view["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
                # Only reset gate states when scene actually resets
                if reset_mask.any():
                    if update_type in ("ttt3r_spectral", "cut3r_spectral",
                                       "cut3r_joint", "ttt3r_joint"):
                        spectral_state = {
                            'ema': state_feat.clone(),
                            'running_energy': torch.zeros_like(
                                spectral_state['running_energy']),
                        }
                    if update_type == "ttt3r_l2gate":
                        l2_state = {
                            'running_energy': torch.zeros_like(
                                l2_state['running_energy']),
                        }
                    if update_type in ("ttt3r_momentum", "ttt3r_brake_geo"):
                        momentum_state = {}
                    if update_type in ("cut3r_geogate", "ttt3r_geogate",
                                       "cut3r_joint", "ttt3r_joint",
                                       "ttt3r_brake_geo"):
                        geo_state = {'prev_depth': curr_depth.detach().clone()}

        if ret_state:
            return ress, views, all_state_args
        return ress, views

    def forward_recurrent_analysis(self, views, device='cuda'):
        """
        Analysis mode inference for Experiment 1: State Token Frequency Visualization.

        Identical inference logic to forward_recurrent_lighter, but additionally captures
        per-frame state token trajectories and state-to-image cross-attention maps.

        Args:
            views:  list of view dicts (same format as forward_recurrent_lighter)
            device: compute device string

        Returns:
            ress:          list of prediction dicts (same as forward_recurrent_lighter)
            analysis_data: dict with keys
                'state_history'      – list of T cpu tensors [n_state, dec_dim],
                                       state BEFORE the update at each frame
                'cross_attn_history' – list of T cpu tensors [n_state, n_img_patches],
                                       mean cross-attention over all decoder layers/heads
                'img_shapes'         – list of T (H_patches, W_patches) tuples
        """
        ress = []
        state_history = []
        cross_attn_history = []
        img_shapes_list = []
        cosine_history = []       # per-frame cosine similarity (mean over tokens)
        gate_history = []         # per-frame gate value (mean over tokens)
        delta_norm_history = []   # per-frame ||delta|| (mean over tokens)
        reset_mask = False
        spectral_state = None

        for i, _view in enumerate(views):
            view = to_gpu(_view, device)
            device = view["img"].device
            batch_size = view["img"].shape[0]

            img_mask = view["img_mask"].reshape(-1, batch_size)
            ray_mask = view["ray_mask"].reshape(-1, batch_size)
            imgs = view["img"].unsqueeze(0)
            ray_maps = view["ray_map"].unsqueeze(0)
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=device)
                    .unsqueeze(0).repeat(batch_size, 1).unsqueeze(0)
            )
            imgs = imgs.view(-1, *imgs.shape[2:])
            ray_maps = ray_maps.view(-1, *ray_maps.shape[2:])
            shapes = shapes.view(-1, 2).to(imgs.device)
            img_masks_flat = img_mask.view(-1)
            ray_masks_flat = ray_mask.view(-1)

            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None

            ray_maps = ray_maps.permute(0, 3, 1, 2)
            selected_ray_maps = ray_maps[ray_masks_flat]
            selected_shapes_ray = shapes[ray_masks_flat]
            if selected_ray_maps.size(0) > 0:
                ray_out, ray_pos, _ = self._encode_ray_map(selected_ray_maps, selected_shapes_ray)
            else:
                ray_out, ray_pos = None, None

            shape = shapes
            if img_out is not None and ray_out is None:
                feat_i = img_out[-1]
                pos_i = img_pos
            elif img_out is None and ray_out is not None:
                feat_i = ray_out[-1]
                pos_i = ray_pos
            elif img_out is not None and ray_out is not None:
                feat_i = img_out[-1] + ray_out[-1]
                pos_i = img_pos
            else:
                raise NotImplementedError

            if i == 0:
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0 or reset_mask:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
                global_img_feat_i = self._get_img_level_feat(feat_i)

            # ── ANALYSIS: record state BEFORE update ──────────────────────────────
            state_history.append(state_feat[0].detach().cpu())  # [n_state, dec_dim]

            new_state_feat, dec, _, cross_attn_state_raw, _, _ = self._recurrent_rollout(
                state_feat, state_pos, feat_i, pos_i,
                pose_feat_i, pose_pos_i, init_state_feat,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update", None),
                return_attn=True,
            )

            # ── ANALYSIS: aggregate cross-attention → [n_state, n_img_patches] ───
            cross_attn_list = list(cross_attn_state_raw)  # list of [1, n_heads, n_state, n_img]
            if len(cross_attn_list) > 0 and cross_attn_list[0] is not None:
                # [n_layers, n_heads, n_state, n_img]
                cross_attn_stacked = torch.cat(cross_attn_list, dim=0)
                # blocks.py returns raw logits (attn_before_softmax); apply softmax before averaging
                cross_attn_stacked = torch.softmax(cross_attn_stacked, dim=-1)
                # average over layers and heads → [n_state, n_img]
                cross_attn_avg = cross_attn_stacked.mean(dim=(0, 1))
                # remove pose token (first column) if pose head is active
                if self.pose_head_flag:
                    cross_attn_img = cross_attn_avg[:, 1:]   # [n_state, n_img_patches]
                else:
                    cross_attn_img = cross_attn_avg
                cross_attn_history.append(cross_attn_img.detach().cpu())

                # patch grid shape (H_p, W_p) derived from actual image height/width
                H_img = int(shapes[0, 0].item())
                W_img = int(shapes[0, 1].item())
                patch_size = 16
                img_shapes_list.append((H_img // patch_size, W_img // patch_size))
            # ──────────────────────────────────────────────────────────────────────

            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)

            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape, pos=pos_i)
            ress.append(to_cpu(res))

            img_mask_val = view["img_mask"]
            update_val = view.get("update", None)
            update_mask = (img_mask_val & update_val) if update_val is not None else img_mask_val
            update_mask = update_mask[:, None, None].float()

            update_type = self.config.model_update_type
            if i == 0 or reset_mask:
                update_mask1 = update_mask
                if update_type in ("ttt3r_spectral", "cut3r_spectral",
                                   "cut3r_joint", "ttt3r_joint"):
                    spectral_state = {
                        'ema': state_feat.clone(),
                        'running_energy': torch.zeros(
                            1, state_feat.shape[1], 1,
                            device=state_feat.device),
                    }
                if update_type in ("cut3r_geogate", "ttt3r_geogate",
                                   "cut3r_joint", "ttt3r_joint",
                                   "ttt3r_brake_geo"):
                    curr_depth = res['pts3d_in_self_view'][0, :, :, 2]
                    geo_state = {'prev_depth': curr_depth.detach().clone()}
                if update_type == "ttt3r_l2gate":
                    l2_state = {
                        'running_energy': torch.zeros(
                            1, state_feat.shape[1], 1,
                            device=state_feat.device),
                    }
                if update_type == "ttt3r_momentum":
                    momentum_state = {}
            else:
                # Extract depth for geo gate types
                if update_type in ("cut3r_geogate", "ttt3r_geogate",
                                   "cut3r_joint", "ttt3r_joint",
                                   "ttt3r_brake_geo"):
                    curr_depth = res['pts3d_in_self_view'][0, :, :, 2]

                if update_type == "cut3r":
                    update_mask1 = update_mask
                elif update_type == "ttt3r":
                    cross_attn_rearr = rearrange(
                        torch.cat(list(cross_attn_state_raw), dim=0),
                        'l h nstate nimg -> 1 nstate nimg (l h)'
                    )
                    state_query_img_key = cross_attn_rearr.mean(dim=(-1, -2))
                    update_mask1 = update_mask * torch.sigmoid(state_query_img_key)[..., None]
                elif update_type == "ttt3r_conf":
                    cross_attn_rearr = rearrange(
                        torch.cat(list(cross_attn_state_raw), dim=0),
                        'l h nstate nimg -> 1 nstate nimg (l h)'
                    )
                    state_query_img_key = cross_attn_rearr.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    conf_scale = getattr(self.config, 'conf_gate_scale', 10.0)
                    if "conf_self" in res:
                        mean_conf = res["conf_self"].mean()
                    elif "conf" in res:
                        mean_conf = res["conf"].mean()
                    else:
                        mean_conf = torch.tensor(conf_scale, device=device)
                    conf_gate = torch.clamp(mean_conf / conf_scale, 0.0, 1.0)
                    update_mask1 = update_mask * ttt3r_mask * conf_gate
                elif update_type == "ttt3r_l2gate":
                    cross_attn_rearr = rearrange(
                        torch.cat(list(cross_attn_state_raw), dim=0),
                        'l h nstate nimg -> 1 nstate nimg (l h)'
                    )
                    state_query_img_key = cross_attn_rearr.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    alpha = self._l2_norm_gate(
                        state_feat, new_state_feat, l2_state, self.config)
                    update_mask1 = update_mask * ttt3r_mask * alpha
                elif update_type == "ttt3r_random":
                    cross_attn_rearr = rearrange(
                        torch.cat(list(cross_attn_state_raw), dim=0),
                        'l h nstate nimg -> 1 nstate nimg (l h)'
                    )
                    state_query_img_key = cross_attn_rearr.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    random_p = getattr(self.config, 'random_gate_p', 0.5)
                    update_mask1 = update_mask * ttt3r_mask * random_p
                elif update_type == "ttt3r_momentum":
                    cross_attn_rearr = rearrange(
                        torch.cat(list(cross_attn_state_raw), dim=0),
                        'l h nstate nimg -> 1 nstate nimg (l h)'
                    )
                    state_query_img_key = cross_attn_rearr.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    m_gate = self._momentum_gate(
                        state_feat, new_state_feat, momentum_state, self.config)
                    update_mask1 = update_mask * ttt3r_mask * m_gate
                elif update_type == "ttt3r_brake_geo":
                    cross_attn_rearr = rearrange(
                        torch.cat(list(cross_attn_state_raw), dim=0),
                        'l h nstate nimg -> 1 nstate nimg (l h)'
                    )
                    state_query_img_key = cross_attn_rearr.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    m_gate = self._momentum_gate(
                        state_feat, new_state_feat, momentum_state, self.config)
                    g_geo = self._geo_consistency_gate(curr_depth, geo_state, self.config)
                    update_mask1 = update_mask * ttt3r_mask * m_gate * g_geo
                elif update_type == "cut3r_spectral":
                    alpha = self._spectral_modulation(
                        state_feat, new_state_feat, spectral_state, self.config)
                    update_mask1 = update_mask * alpha
                elif update_type == "ttt3r_spectral":
                    cross_attn_rearr = rearrange(
                        torch.cat(list(cross_attn_state_raw), dim=0),
                        'l h nstate nimg -> 1 nstate nimg (l h)'
                    )
                    state_query_img_key = cross_attn_rearr.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    alpha = self._spectral_modulation(
                        state_feat, new_state_feat, spectral_state, self.config)
                    update_mask1 = update_mask * ttt3r_mask * alpha
                elif update_type == "cut3r_geogate":
                    g_geo = self._geo_consistency_gate(curr_depth, geo_state, self.config)
                    update_mask1 = update_mask * g_geo
                elif update_type == "ttt3r_geogate":
                    cross_attn_rearr = rearrange(
                        torch.cat(list(cross_attn_state_raw), dim=0),
                        'l h nstate nimg -> 1 nstate nimg (l h)'
                    )
                    state_query_img_key = cross_attn_rearr.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    g_geo = self._geo_consistency_gate(curr_depth, geo_state, self.config)
                    update_mask1 = update_mask * ttt3r_mask * g_geo
                elif update_type == "cut3r_joint":
                    alpha = self._spectral_modulation(
                        state_feat, new_state_feat, spectral_state, self.config)
                    g_geo = self._geo_consistency_gate(curr_depth, geo_state, self.config)
                    update_mask1 = update_mask * alpha * g_geo
                elif update_type == "ttt3r_joint":
                    cross_attn_rearr = rearrange(
                        torch.cat(list(cross_attn_state_raw), dim=0),
                        'l h nstate nimg -> 1 nstate nimg (l h)'
                    )
                    state_query_img_key = cross_attn_rearr.mean(dim=(-1, -2))
                    ttt3r_mask = torch.sigmoid(state_query_img_key)[..., None]
                    alpha = self._spectral_modulation(
                        state_feat, new_state_feat, spectral_state, self.config)
                    g_geo = self._geo_consistency_gate(curr_depth, geo_state, self.config)
                    update_mask1 = update_mask * ttt3r_mask * alpha * g_geo
                else:
                    raise ValueError(f"Invalid model type: {update_type}")

            # Log cosine similarity, gate value, and delta norm for analysis
            delta = new_state_feat - state_feat  # [1, n_state, D]
            delta_norm_history.append(delta.detach().norm(dim=-1).mean().cpu().item())
            if len(state_history) >= 2:
                prev_delta_vec = state_history[-1] - (state_history[-2] if len(state_history) >= 2 else state_history[-1])
                # Recompute from raw deltas stored in a running buffer
                pass  # cosine computed below from prev_raw_delta
            if not hasattr(self, '_analysis_prev_delta') or i == 0:
                self._analysis_prev_delta = delta.detach().clone()
                cosine_history.append(0.0)
                gate_history.append(0.5)
            else:
                cos_val = torch.nn.functional.cosine_similarity(
                    delta, self._analysis_prev_delta, dim=-1
                ).mean().cpu().item()
                cosine_history.append(cos_val)
                tau = getattr(self.config, 'momentum_tau', 2.0)
                gate_history.append(torch.sigmoid(torch.tensor(-tau * cos_val)).item())
                self._analysis_prev_delta = delta.detach().clone()

            state_feat = new_state_feat * update_mask1 + state_feat * (1 - update_mask1)
            mem = new_mem * update_mask + mem * (1 - update_mask)

            reset_mask = view["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
                # Only reset gate states when scene actually resets
                if reset_mask.any():
                    if update_type in ("ttt3r_spectral", "cut3r_spectral",
                                       "cut3r_joint", "ttt3r_joint"):
                        spectral_state = {
                            'ema': state_feat.clone(),
                            'running_energy': torch.zeros_like(
                                spectral_state['running_energy']),
                        }
                    if update_type == "ttt3r_l2gate":
                        l2_state = {
                            'running_energy': torch.zeros_like(
                                l2_state['running_energy']),
                        }
                    if update_type in ("ttt3r_momentum", "ttt3r_brake_geo"):
                        momentum_state = {}

        # Clean up temporary state
        if hasattr(self, '_analysis_prev_delta'):
            del self._analysis_prev_delta

        analysis_data = {
            'state_history': state_history,        # list[T] of [n_state, dec_dim]
            'cross_attn_history': cross_attn_history,  # list[T] of [n_state, n_img_patches]
            'img_shapes': img_shapes_list,         # list[T] of (H_patches, W_patches)
            'cosine_history': cosine_history,      # list[T] of float, mean cosine sim
            'gate_history': gate_history,           # list[T] of float, sigmoid(-tau*cos)
            'delta_norm_history': delta_norm_history,  # list[T] of float, mean ||delta||
        }
        return ress, analysis_data

if __name__ == "__main__":
    print(ARCroco3DStereo.mro())
    cfg = ARCroco3DStereoConfig(
        state_size=256,
        pos_embed="RoPE100",
        rgb_head=True,
        pose_head=True,
        img_size=(224, 224),
        head_type="linear",
        output_mode="pts3d+pose",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        pose_mode=("exp", -inf, inf),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
    )
    ARCroco3DStereo(cfg)
