# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .base_entity import BaseEntity


class BboxEntity(BaseEntity):

    COMMAND_STRING = "make_bbox"

    PARAMS_DEFINITION = {
        "id": {"dtype": int, "normalise": None},
        "class": {"dtype": str, "normalise": "bbox_classes"},
        "position_x": {"dtype": float, "normalise": "world"},
        "position_y": {"dtype": float, "normalise": "world"},
        "position_z": {"dtype": float, "normalise": "world"},
        "angle_z": {"dtype": float, "normalise": "angle"},
        "scale_x": {"dtype": float, "normalise": "scale"},
        "scale_y": {"dtype": float, "normalise": "scale"},
        "scale_z": {"dtype": float, "normalise": "scale"},
    }

    TOKEN = 4

    def __init__(self, parameters):
        """
        Args:
            parameters: Dict with keys specified in BboxEntity.PARAMS_DEFINITION.
        """
        re_ordered_parameters = {}
        for param_key in BboxEntity.PARAMS_DEFINITION:
            if param_key in parameters:
                re_ordered_parameters[param_key] = parameters[param_key]
        self.params = re_ordered_parameters

    def extent(self):
        """Compute extent of bbox.

        Returns:
            Dict with the following keys: {min/max/size}_{x/y/z}.
                Values are floats.
        """
        canonical_bbox = (
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                ]
            )
            - 0.5
        )  # [8, 3]. Centered at [0, 0, 0], sidelength of 1

        # Scale it
        scales = np.array(
            [self.params["scale_x"], self.params["scale_y"], self.params["scale_z"]]
        )
        bbox = canonical_bbox * scales[None]

        # Rotate it
        rot_mat = Rotation.from_euler("ZYX", [self.params["angle_z"], 0, 0]).as_matrix()
        bbox = bbox @ rot_mat.T

        # Translate it
        translation = np.array(
            [
                self.params["position_x"],
                self.params["position_y"],
                self.params["position_z"],
            ]
        )
        bbox = bbox + translation

        min_x = bbox[:, 0].min()
        max_x = bbox[:, 0].max()
        min_y = bbox[:, 1].min()
        max_y = bbox[:, 1].max()
        min_z = bbox[:, 2].min()
        max_z = bbox[:, 2].max()

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
        """Rotate bbox entity.

        Args:
            rotation_angle: float. Angle to rotate in degrees about the Z-axis.
        """
        augment_rot_mat = Rotation.from_euler(
            "ZYX", [rotation_angle, 0, 0], degrees=True
        ).as_matrix()

        # Rotate the rotation
        bbox_rot_mat = Rotation.from_euler(
            "ZYX", [self.params["angle_z"], 0, 0]
        ).as_matrix()
        new_bbox_rot_mat = augment_rot_mat @ bbox_rot_mat
        new_angle_z = Rotation.from_matrix(new_bbox_rot_mat).as_euler("ZYX")[0]
        new_angle_z = (new_angle_z + np.pi) % (2 * np.pi) - np.pi  # Range: [-pi, pi)

        # Bbox is symmetric
        symmetry = np.pi
        if np.isclose(self.params["scale_x"], self.params["scale_y"], atol=1e-3):
            symmetry = np.pi / 2
        new_angle_z = (new_angle_z + np.pi) % symmetry - np.pi
        # Note: will modulo angle_z into [-pi, 0) or [-pi, -pi/2) depending on symmetry assumption
        self.params["angle_z"] = new_angle_z

        bbox_center = np.array(
            [
                self.params["position_x"],
                self.params["position_y"],
                self.params["position_z"],
            ]
        )
        new_bbox_center = augment_rot_mat @ bbox_center
        self.params["position_x"] = new_bbox_center[0]
        self.params["position_y"] = new_bbox_center[1]
        self.params["position_z"] = new_bbox_center[2]

    def translate(self, translation):
        """Translate bbox entity.

        Args:
            translation: [3] np.ndarray of XYZ translation.
        """
        bbox_center = torch.as_tensor(
            [
                self.params["position_x"],
                self.params["position_y"],
                self.params["position_z"],
            ],
        ).float()
        new_bbox_center = bbox_center + translation

        self.params["position_x"] = float(new_bbox_center[0])
        self.params["position_y"] = float(new_bbox_center[1])
        self.params["position_z"] = float(new_bbox_center[2])

    def lex_sort_key(self):
        """Compute sorting key for lexicographic sorting.

        Returns:
            a [2] np.ndarray.
        """
        bbox_center_xy = np.array(
            [
                self.params["position_x"],
                self.params["position_y"],
            ]
        )
        return bbox_center_xy

    def random_sort_key(self):
        """Compute sorting key for random sorting.

        Returns:
            a [1] np.ndarray.
        """
        return np.random.rand(1)  # [1]
