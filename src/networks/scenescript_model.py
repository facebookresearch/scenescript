# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
from enum import IntEnum

import torch
import torchsparse
from omegaconf import OmegaConf

from src.data.geometries import ALL_ENTITY_CLASSES, get_entity_class_from_token
from src.data.language_sequence import is_id_param, LanguageSequence
from src.data.point_cloud import PointCloud
from src.networks.decoder import HELPER_TOKEN, SceneScriptDecoder
from src.networks.encoder import PointCloudEncoder
from torch.nn import functional as F, modules as nn
from torchsparse.utils.collate import sparse_collate


def create_TYPE_TOKEN():
    values = ["PAD", "START", "STOP", "PART", "NOT_USED", "NOT_USED_1", "COMMAND"]
    for ENTITY_CLASS in ALL_ENTITY_CLASSES:
        for param_key in ENTITY_CLASS.PARAMS_DEFINITION:
            values.append("_".join([ENTITY_CLASS.COMMAND_STRING, param_key]).upper())
    values.append("NUM")
    return IntEnum("TYPE_TOKEN", values, start=0)


def list_rindex(list, value):
    list.reverse()
    i = list.index(value)
    list.reverse()
    return len(list) - i - 1


class SceneScriptWrapper(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_num_tokens = cfg.model.decoder.max_num_tokens
        self.type_token = create_TYPE_TOKEN()

        self.encoder = PointCloudEncoder(
            input_channels=cfg.model.encoder.input_channels,
            d_model=cfg.model.encoder.d_model,
            conv_layers=cfg.model.encoder.conv_layers,
            num_bins=cfg.model.encoder.num_bins,
        )

        self.decoder = SceneScriptDecoder(
            d_model=cfg.model.decoder.d_model,
            num_attn_heads=cfg.model.decoder.num_attn_heads,
            dim_feedforward=cfg.model.decoder.dim_feedforward,
            num_bins=cfg.model.decoder.num_bins,
            max_num_tokens=cfg.model.decoder.max_num_tokens,
            max_num_type_tokens=self.type_token.NUM,
            num_decoder_layers=cfg.model.decoder.num_decoder_layers,
        )

        self.model = nn.ModuleDict({"encoder": self.encoder, "decoder": self.decoder})

    @staticmethod
    def load_from_checkpoint(ckpt_path):
        ckpt_dict = torch.load(ckpt_path)
        cfg = OmegaConf.create(ckpt_dict["cfg"])

        model_wrapper = SceneScriptWrapper(cfg)
        model_wrapper.model.load_state_dict(ckpt_dict["model_state_dict"])
        model_wrapper.model = model_wrapper.model.eval()

        return model_wrapper

    @property
    def device(self):
        return list(self.model.parameters())[0].device

    def cuda(self):
        self.model = self.model.cuda()
        return self

    def top_p(self, logits, thres):
        """Filter out logits for nucleus sampling.

        Args:
            logits: [B, num_bins + HELPER_TOKEN.NUM] torch.Tensor.
            thresh: float. 0 means argmax, 1 means random sampling.

        Returns:
            filtered_logits: [B, num_bins + HELPER_TOKEN.NUM] torch.Tensor.
        """
        # Sort the logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cum_probs >= thres

        # Include the bin that pushed cumulative probability above 1 - thresh
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        # Set filtered logits to -inf, effectively ignoring them
        sorted_logits[sorted_indices_to_remove] = float("-inf")

        # Scatter will put logits back in their places
        return sorted_logits.scatter(1, sorted_indices, sorted_logits)

    def type_decoding(self, seq_value, seq_type):
        """Decode the next type token.

        Args:
            seq_value: [B, t] torch.LongTensor.
            seq_type: [B, t-1] torch.LongTensor.

        Returns:
            [B, t] torch.LongTensor.
        """
        new_type = torch.zeros_like(seq_type[:, 0])  # [B]
        for b in range(seq_value.shape[0]):
            seq_value_b = seq_value[b]  # [t]
            seq_type_b = seq_type[b]  # [t - 1]

            try:
                # There's already a stop token
                if torch.any(seq_type_b == self.type_token.STOP):
                    new_type[b] = self.type_token.PAD

                # A PART token was predicted in sequence
                elif seq_value_b[-1] == HELPER_TOKEN.PART:
                    new_type[b] = self.type_token.PART

                # A STOP token was predicted in sequence
                elif seq_value_b[-1] == HELPER_TOKEN.STOP:
                    new_type[b] = self.type_token.STOP

                # Previously, a COMMAND token was predicted
                elif seq_type_b[-1] == self.type_token.PART:
                    new_type[b] = self.type_token.COMMAND

                # We are somewhere in the middle of an argument sequence
                else:
                    latest_command_token_idx = list_rindex(
                        seq_type_b.tolist(), self.type_token.COMMAND
                    )
                    command_value = int(
                        seq_value_b[latest_command_token_idx] - HELPER_TOKEN.NUM
                    )
                    ENTITY_CLASS = get_entity_class_from_token(command_value)

                    type_token_ordering = (
                        [self.type_token.COMMAND]
                        + [
                            self.type_token[
                                f"{ENTITY_CLASS.COMMAND_STRING}_{param_key}".upper()
                            ]
                            for param_key in ENTITY_CLASS.PARAMS_DEFINITION
                            if not is_id_param(param_key)
                        ]
                    )  # e.g. [COMMAND, MAKE_WALL_A_X, MAKE_WALL_A_Y, ..., MAKE_WALL_HEIGHT]

                    token_order_idx = type_token_ordering.index(seq_type_b[-1])
                    new_type[b] = type_token_ordering[token_order_idx + 1]

            except:  # for any errors, just pad
                new_type[b] = self.type_token.PAD

        return torch.cat([seq_type, new_type.unsqueeze(-1)], dim=-1)

    def preprocess_point_cloud(self, point_cloud):
        """Preprocess the point cloud to be fed into the encoder.

        Args:
            point_cloud: [N, 3] torch.FloatTensor.

        Returns:
            sparse_tensor: torchsparse.SparseTensor.
        """
        point_cloud = PointCloud(point_cloud)

        # Push to positive quadrant
        extent = point_cloud.extent()
        pc_min = [extent["min_x"], extent["min_y"], extent["min_z"]]
        pc_min = torch.as_tensor(pc_min)
        point_cloud.translate(-pc_min)

        # Normalize / Discretize it
        point_cloud.normalize_and_discretize(
            self.cfg.data.num_bins, self.cfg.data.normalization_values
        )

        # Convert to torchsparse.SparseTensor
        pc_sparse_tensor = torchsparse.SparseTensor(
            coords=point_cloud.coords.int(),
            feats=point_cloud.points.float(),
        )

        pc_sparse_tensor = sparse_collate([pc_sparse_tensor])  # batch_size = 1
        pc_sparse_tensor = pc_sparse_tensor.to(self.device)

        return pc_sparse_tensor, pc_min

    def postprocess_language(self, seq_value, pc_min):
        """Postprocess the language sequence back into the original frame of reference.

        Args:
            seq_value: [T] torch.LongTensor.
            pc_min: [3] torch.FloatTensor.
        """
        language_sequence = LanguageSequence.from_seq_value(seq_value)
        language_sequence.undiscretize_and_unnormalize(
            self.cfg.data.num_bins, self.cfg.data.normalization_values
        )
        language_sequence.translate(pc_min)

        return language_sequence

    @torch.no_grad()
    def run_inference(
        self,
        raw_point_cloud,
        nucleus_sampling_thresh=0.05,
        verbose=False,
    ):
        """Run the full inference loop.

        Args:
            raw_point_cloud: [N, 3] torch.FloatTensor.
            nucleus_sampling_thresh: float. In [0, 1]. 0 means argmax, 1 means random sampling.
            verbose: bool.

        Returns:
            a LanguageSequence instance.
        """
        start_time = time.time()

        # Encode the visual inputs
        pc_sparse_tensor, pc_min = self.preprocess_point_cloud(raw_point_cloud)
        encoded_visual_input = self.model["encoder"](pc_sparse_tensor)
        context = encoded_visual_input["context"]
        context_mask = encoded_visual_input["context_mask"]

        if verbose:
            print(f"Time taken for input encoding: {time.time() - start_time:.3f}s")
            start_time = time.time()  # reset timer

        B = context.shape[0]
        device = self.device

        seq_value = (
            torch.ones((B, 1), dtype=torch.long, device=device) * HELPER_TOKEN.START
        )
        seq_type = (
            torch.ones((B, 1), dtype=torch.long, device=device) * self.type_token.START
        )

        for _ in range(seq_value.shape[1], self.max_num_tokens):
            # Run decoder to get logits
            logits = self.model["decoder"](
                context=context,
                context_mask=context_mask,
                seq_value=seq_value,
                seq_type=seq_type,
            )  # [B, T, num_bins + HELPER_TOKEN.NUM]
            logits_t = logits[:, -1]  # [B, num_bins + HELPER_TOKEN.NUM]
            logits_filtered = self.top_p(logits_t, nucleus_sampling_thresh)

            # Sample a token (across batch)
            probs = F.softmax(logits_filtered, dim=-1)  # [B, ...]
            tokens = torch.multinomial(probs, 1)  # [B, 1]

            # Append token
            seq_value = torch.cat([seq_value, tokens], dim=1)  # [B, t+1]

            # Decode type token
            seq_type = self.type_decoding(seq_value, seq_type)

            # Stop if the sequence has a STOP token
            if torch.sum(seq_value[0] == HELPER_TOKEN.STOP) >= 1:
                break

        if verbose:
            print(
                f"Time taken for autoregressive sampling: {time.time() - start_time:.3f}s"
            )

        seq_value = seq_value[0]  # un-batch-ify
        language_sequence = self.postprocess_language(seq_value, pc_min)

        return language_sequence
