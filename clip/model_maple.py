"""CLIP model adapter required by MaPLe's deep multimodal prompts."""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from .model import LayerNorm, ModifiedResNet, QuickGELU, convert_weights


class ResidualAttentionBlockMaPLe(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        attn_mask=None,
        design_details=None,
        text_layer=False,
        layer_index=0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        self.compound_prompt_nctx = design_details["maple_length"]
        self.first_layer = layer_index == 0

    def attention(self, x):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        return self.attn(
            x, x, x, need_weights=False, attn_mask=self.attn_mask
        )[0]

    def forward(self, inputs):
        x, compound_prompts, counter = inputs
        if not self.first_layer and counter < len(compound_prompts):
            context = compound_prompts[counter]
            context = context.expand(x.shape[1], -1, -1)
            context = context.permute(1, 0, 2).to(dtype=x.dtype)
            if self.text_layer:
                prefix = x[:1]
                suffix = x[1 + self.compound_prompt_nctx :]
                x = torch.cat([prefix, context, suffix], dim=0)
            else:
                prefix = x[: x.shape[0] - self.compound_prompt_nctx]
                x = torch.cat([prefix, context], dim=0)
            counter += 1

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return [x, compound_prompts, counter]


class TransformerMaPLe(nn.Module):
    def __init__(
        self,
        width,
        layers,
        heads,
        attn_mask=None,
        text_layer=False,
        design_details=None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlockMaPLe(
                    width,
                    heads,
                    attn_mask,
                    design_details,
                    text_layer,
                    layer_index,
                )
                for layer_index in range(layers)
            ]
        )

    def forward(self, inputs):
        return self.resblocks(inputs)


class VisionTransformerMaPLe(nn.Module):
    def __init__(
        self,
        input_resolution,
        patch_size,
        width,
        layers,
        heads,
        output_dim,
        design_details,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            3, width, kernel_size=patch_size, stride=patch_size, bias=False
        )
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        grid = input_resolution // patch_size
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(grid * grid + 1, width)
        )
        self.ln_pre = LayerNorm(width)
        self.transformer = TransformerMaPLe(
            width, layers, heads, design_details=design_details
        )
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x, shared_ctx, compound_deeper_prompts):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_token = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([class_token, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).to(dtype=x.dtype)
        x = torch.cat([x, visual_ctx], dim=1)
        x = self.ln_pre(x).permute(1, 0, 2)
        x = self.transformer([x, compound_deeper_prompts, 0])[0]
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


class CLIPMaPLe(nn.Module):
    def __init__(
        self,
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        design_details,
    ):
        super().__init__()
        self.context_length = context_length
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformerMaPLe(
                image_resolution,
                vision_patch_size,
                vision_width,
                vision_layers,
                vision_heads,
                embed_dim,
                design_details,
            )

        self.transformer = TransformerMaPLe(
            transformer_width,
            transformer_layers,
            transformer_heads,
            attn_mask=self.build_attention_mask(),
            text_layer=True,
            design_details=design_details,
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
            for resnet_block in [
                self.visual.layer1,
                self.visual.layer2,
                self.visual.layer3,
                self.visual.layer4,
            ]:
                for name, parameter in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(parameter)

        proj_std = (self.transformer.width ** -0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype


def build_model_MaPLe(state_dict, design_details):
    if "visual.proj" not in state_dict:
        raise ValueError("MaPLe requires a Vision Transformer CLIP backbone")

    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [
            key
            for key in state_dict
            if key.startswith("visual.")
            and key.endswith(".attn.in_proj_weight")
        ]
    )
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round(
        (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5
    )
    image_resolution = vision_patch_size * grid_size
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            key.split(".")[2]
            for key in state_dict
            if key.startswith("transformer.resblocks")
        )
    )

    model = CLIPMaPLe(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        design_details,
    )
    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)
    convert_weights(model)
    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict, strict=False
    )
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "MaPLe CLIP state mismatch: missing={} unexpected={}".format(
                missing_keys, unexpected_keys
            )
        )
    return model.eval()
