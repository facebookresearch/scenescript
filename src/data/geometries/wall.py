# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .base_entity import BaseEntity


class WallEntity(BaseEntity):

    COMMAND_STRING = "make_wall"

    PARAMS_DEFINITION = {
        "id": {"dtype": int, "normalise": None},
        "a_x": {"dtype": float, "normalise": "world"},
        "a_y": {"dtype": float, "normalise": "world"},
        "a_z": {"dtype": float, "normalise": "world"},
        "b_x": {"dtype": float, "normalise": "world"},
        "b_y": {"dtype": float, "normalise": "world"},
        "b_z": {"dtype": float, "normalise": "world"},
        "height": {"dtype": float, "normalise": "height"},
        "thickness": {"dtype": float, "normalise": "height"},
        # Note: thickness is always 0... this is a legacy parameter
    }

    TOKEN = 1

    def __init__(self, parameters):
        """
        Args:
            parameters: Dict with keys specified in WallEntity.PARAMS_DEFINITION.
        """
        re_ordered_parameters = {}
        for param_key in WallEntity.PARAMS_DEFINITION:
            if param_key in parameters:
                re_ordered_parameters[param_key] = parameters[param_key]
        self.params = re_ordered_parameters

    def extent(self):
        """Compute extent of wall.

        Returns:
            Dict with the following keys: {min/max/size}_{x/y/z}.
                Values are floats.
        """
        min_x = min(self.params["a_x"], self.params["b_x"])
        max_x = max(self.params["a_x"], self.params["b_x"])
        min_y = min(self.params["a_y"], self.params["b_y"])
        max_y = max(self.params["a_y"], self.params["b_y"])
        min_z = min(self.params["a_z"], self.params["b_z"])
        max_z = max(self.params["a_z"], self.params["b_z"]) + self.params["height"]
        return {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "min_z": min_z,
            "max_z": max_z,
            "size_x": max(max_x - min_x, 0),
            "size_y": max(max_y - min_y, 0),
            "size_z": max(max_z - min_z, 0),
        }

    def rotate(self, rotation_angle):
        """Rotate wall entity.

        Args:
            rotation_angle: float. Angle to rotate in degrees about the Z-axis.
        """
        wall_start = np.array(
            [self.params["a_x"], self.params["a_y"], self.params["a_z"]]
        )
        wall_end = np.array(
            [self.params["b_x"], self.params["b_y"], self.params["b_z"]]
        )

        rot_mat = Rotation.from_euler(
            "ZYX", [rotation_angle, 0, 0], degrees=True
        ).as_matrix()
        new_wall_start = rot_mat @ wall_start
        new_wall_end = rot_mat @ wall_end

        self.params["a_x"] = new_wall_start[0]
        self.params["a_y"] = new_wall_start[1]
        self.params["a_z"] = new_wall_start[2]
        self.params["b_x"] = new_wall_end[0]
        self.params["b_y"] = new_wall_end[1]
        self.params["b_z"] = new_wall_end[2]

    def translate(self, translation):
        """Translate wall entity.

        Args:
            translation: [3] np.ndarray of XYZ translation.
        """
        wall_start = torch.as_tensor(
            [self.params["a_x"], self.params["a_y"], self.params["a_z"]],
        ).float()
        wall_end = torch.as_tensor(
            [self.params["b_x"], self.params["b_y"], self.params["b_z"]],
        ).float()

        new_wall_start = wall_start + translation
        new_wall_end = wall_end + translation

        self.params["a_x"] = new_wall_start[0]
        self.params["a_y"] = new_wall_start[1]
        self.params["a_z"] = new_wall_start[2]
        self.params["b_x"] = new_wall_end[0]
        self.params["b_y"] = new_wall_end[1]
        self.params["b_z"] = new_wall_end[2]

    def lex_sort_key(self):
        """Compute sorting key for lexicographic sorting.

        Note: self.params will be edited with corner lex-sorting as well.

        Returns:
            a [6] np.ndarray.
        """

        # Lex-sort corners
        wall_start = np.array(
            [self.params["a_x"], self.params["a_y"], self.params["a_z"]]
        )
        wall_end = np.array(
            [self.params["b_x"], self.params["b_y"], self.params["b_z"]]
        )
        corners = np.stack([wall_start, wall_end])  # [2, 3]

        idx = np.lexsort(corners.T)  # [2]. Sorts by z, y, x.
        corner_1_ordered, corner_2_ordered = corners[idx]

        # Sort wall-corners
        self.params["a_x"], self.params["a_y"], self.params["a_z"] = corner_1_ordered
        self.params["b_x"], self.params["b_y"], self.params["b_z"] = corner_2_ordered

        return np.concatenate([corner_2_ordered, corner_1_ordered])
