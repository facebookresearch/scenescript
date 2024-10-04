# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd

import torch


class PointCloud:
    def __init__(self, points):
        """A class that wraps some point cloud functionality.

        Args:
            points: [N, 3] torch.FloatTensor of XYZ coordinates of the point cloud.
        """
        self.points = points

    @staticmethod
    def load_from_file(
        point_cloud_filename,
        distance_std_threshold=0.01,
        inverse_distance_std_threshold=0.001,
    ):
        """A class that wraps some point cloud functionality.

        Args:
            point_cloud_filename: str. Path to point cloud file output by Aria MPS.
            distance_std_threshold: float. Threshold on the standard deviation of the distance.
            inverse_distance_std_threshold: float. Threshold on the standard deviation of the inverse depth.
        """
        try:
            import projectaria_tools.core.mps as mps
            from projectaria_tools.core.mps.utils import filter_points_from_confidence

            all_points = mps.read_global_point_cloud(point_cloud_filename)

            # filter the point cloud using thresholds on the inverse depth and distance standard deviation
            filtered_points = filter_points_from_confidence(
                all_points, inverse_distance_std_threshold, distance_std_threshold
            )

            # turn into np.array
            points = np.array(
                [point.position_world.astype(np.float32) for point in filtered_points]
            )

        except ImportError:
            print("projectaria_tools not installed")

            with open(point_cloud_filename, "rb") as f:
                points_df = pd.read_csv(f, compression="gzip")
                print(f"Loaded {points_df.shape[0]} points from {point_cloud_filename}")

            # filter the point cloud using thresholds on the inverse depth and distance standard deviation
            points_df = points_df[
                (points_df["inv_dist_std"] <= inverse_distance_std_threshold)
                & (points_df["dist_std"] <= distance_std_threshold)
            ]

            # turn into np.array
            points = points_df[["px_world", "py_world", "pz_world"]].values

        print(f"Kept {points.shape[0]} points after filtering!")

        return PointCloud(points=torch.as_tensor(points).float())

    def extent(self):
        """Compute extent of point cloud.

        Returns:
            Dict with the following keys: {min/max/size}_{x/y/z}.
                Values are floats.
        """

        min_x = 1e6
        min_y = 1e6
        min_z = 1e6
        max_x = -1e6
        max_y = -1e6
        max_z = -1e6

        points_min = self.points.min(dim=0)[0]
        min_x = min(min_x, points_min[0])
        min_y = min(min_y, points_min[1])
        min_z = min(min_z, points_min[2])

        points_max = self.points.max(dim=0)[0]
        max_x = max(max_x, points_max[0])
        max_y = max(max_y, points_max[1])
        max_z = max(max_z, points_max[2])

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

    def translate(self, translation_vector):
        """Translate point cloud.

        Args:
            translation_vector: [3] torch.FloatTensor of XYZ translation vector.
        """
        points_xyz = self.points[:, :3]
        points_rest = self.points[:, 3:]
        translated_points = points_xyz + translation_vector
        self.points = torch.cat([translated_points, points_rest], dim=1)

    def normalize_and_discretize(self, num_bins, normalization_values):
        """Normalize and Discretize the point cloud.

        Args:
            num_bins: int.
            normalization_values: Dict[str, List[Union[float, str]]].
                The keys are strings that can be found in geometries/*.py.
                Examples are: ["world", "width", "height", "scale", "angle"].
                Values can be either List[float] or List[str].
                    List[float] are used for min/max value (e.g. min/max width/height).
                    List[str] is used for categories (e.g. ["table", "chair"]) for "bbox_classes".
                    Examples:
                        "world": [0.0, 32.0],
                        "width": [0.0, 5.0],
                        "bbox_classes": ["table", "chair"],
                        ...
        """
        original_points = self.points.clone()
        normalization_extent = (
            normalization_values["world"][1] - normalization_values["world"][0]
        )

        # Translate to positive quadrant
        positive_quadrant_points = original_points - original_points.min(dim=0)[0]

        # Normalise
        normalized_points = (
            positive_quadrant_points / normalization_extent
        )  # [N, 3]. Range: [0, 1]

        # Discretise
        voxel_coords = (normalized_points * num_bins).round().long()
        voxel_coords = voxel_coords.clamp(max=num_bins - 1)

        # Get unique voxel coordinates
        unique_voxel_coords, inverse, unique_voxel_counts = np.unique(
            voxel_coords.numpy(), axis=0, return_inverse=True, return_counts=True
        )
        unique_voxel_coords = torch.as_tensor(unique_voxel_coords)
        inverse = torch.as_tensor(inverse)
        unique_voxel_counts = torch.as_tensor(unique_voxel_counts)

        # Average of points falling in the same bin
        discretised_original_points = torch.stack(
            [
                torch.bincount(inverse, weights=original_points[:, i])
                / unique_voxel_counts
                for i in range(original_points.shape[1])
            ],
            dim=1,
        )

        self.points = discretised_original_points
        self.coords = unique_voxel_coords  # in {0, 1, ..., num_bins - 1}
