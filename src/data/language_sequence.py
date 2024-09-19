# Copyright (c) Meta Platforms, Inc. and affiliates.
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

import numpy as np
import torch

from src.data.geometries import (
    ALL_ENTITY_CLASSES,
    DoorEntity,
    get_entity_class_from_string,
    get_entity_class_from_token,
    WallEntity,
    WindowEntity,
)
from src.networks.decoder import HELPER_TOKEN


def is_id_param(param):
    return param == "id" or "_id" in param


def point_to_line_seg_dist(point, line_seg):
    """Compute point to line segment distance.

    Args:
        point: [2] np.ndarray.
        line_seg: Tuple[np.ndarray]. This tuple consists of 2 elements:
            np.ndarray of (x1, y1).
            np.ndarray of (x2, y2).

    Returns:
        scalar.
    """
    # unit vector
    unit_line_seg = line_seg[1] - line_seg[0]  # [2]
    norm_unit_line_seg = unit_line_seg / np.linalg.norm(unit_line_seg)  # [2]

    # compute the perpendicular distance to the theoretical infinite line_seg
    segment_dist = np.linalg.norm(
        np.cross(line_seg[1] - line_seg[0], line_seg[0] - point)
    ) / np.linalg.norm(
        unit_line_seg
    )  # scalar

    # Project point to line
    diff = norm_unit_line_seg.dot(point - line_seg[0])
    xy_seg = norm_unit_line_seg * diff + line_seg[0]
    x_seg, y_seg = xy_seg

    # Distance of point to line segment endpoints
    endpoint_dist = min(
        np.linalg.norm(line_seg[0] - point), np.linalg.norm(line_seg[1] - point)
    )

    # decide if the intersection point falls on the line_seg segment
    lp1_x = line_seg[0][0]  # line_seg point 1 x
    lp1_y = line_seg[0][1]  # line_seg point 1 y
    lp2_x = line_seg[1][0]  # line_seg point 2 x
    lp2_y = line_seg[1][1]  # line_seg point 2 y
    is_betw_x = lp1_x <= x_seg <= lp2_x or lp2_x <= x_seg <= lp1_x
    is_betw_y = lp1_y <= y_seg <= lp2_y or lp2_y <= y_seg <= lp1_y
    if is_betw_x and is_betw_y:
        return segment_dist
    else:
        # if not, then return the minimum distance to the segment endpoints
        return endpoint_dist


class LanguageSequence:

    def __init__(self, entities):
        """
        Args:
            entities: List[Tuple[str, Dict]]. Each Tuple[str, Dict] item:
                str is the command (e.g. "make_wall")
                Dict is the parameters (e.g. id=0, a_x=3.459)
        """
        self.entities = entities

    @staticmethod
    def load_from_file(filepath):
        """
        Args:
            filepath: str. Path to local file.
        """
        entities = []

        # Read and load entities
        with open(filepath, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.rstrip()
                entries = line.split(", ")
                command_string = entries[0]

                # Get the correct entity class
                for ENTITY_CLASS in ALL_ENTITY_CLASSES:
                    if ENTITY_CLASS.COMMAND_STRING == command_string:
                        break

                # Get the parameters
                params = {}
                for parameter_def in entries[1:]:
                    key, value = parameter_def.split("=")
                    if key in ENTITY_CLASS.PARAMS_DEFINITION:
                        dtype = ENTITY_CLASS.PARAMS_DEFINITION[key]["dtype"]
                        params[key] = dtype(value)

                # Instantiate entity
                if ENTITY_CLASS.COMMAND_STRING in ["make_door", "make_window"]:
                    parent_wall_id = params["wall0_id"]
                    parent_wall_entity = None
                    for _ent in entities:
                        if _ent.params["id"] == parent_wall_id:
                            parent_wall_entity = _ent
                    assert parent_wall_entity is not None
                    new_entity = ENTITY_CLASS(params, parent_wall_entity)

                else:
                    new_entity = ENTITY_CLASS(params)

                entities.append(new_entity)

        return LanguageSequence(entities)

    def extent(self):
        """Compute extent of language.

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

        for entity in self.entities:
            entity_extent = entity.extent()
            min_x = min(min_x, entity_extent["min_x"])
            max_x = max(max_x, entity_extent["max_x"])
            min_y = min(min_y, entity_extent["min_y"])
            max_y = max(max_y, entity_extent["max_y"])
            min_z = min(min_z, entity_extent["min_z"])
            max_z = max(max_z, entity_extent["max_z"])

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
        """Rotate language entities.

        Args:
            rotation_angle: float. Angle to rotate in degrees about the Z-axis.
        """
        for entity in self.entities:
            entity.rotate(rotation_angle)

    def translate(self, translation):
        """Translate language entities.

        Args:
            translation: [3] torch.FloatTensor of XYZ translation vector.
        """
        for entity in self.entities:
            entity.translate(translation)

    def normalize_and_discretize(self, num_bins, normalization_values):
        """Normalize and discretize language entities.

        Note: Assumes the language has been normalized, i.e. the range is in [0,1].

        Args:
            num_bins: int. Number of bins to discretise to.
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
        for entity in self.entities:
            for key in entity.PARAMS_DEFINITION:

                normalization_strategy = entity.PARAMS_DEFINITION[key]["normalise"]
                dtype = entity.PARAMS_DEFINITION[key]["dtype"]

                if dtype == float:

                    min_val, max_val = normalization_values[normalization_strategy]
                    normalized_value = (entity.params[key] - min_val) / (
                        max_val - min_val
                    )  # scalar. range: [0, 1]

                    rounded_value = int(normalized_value * num_bins)
                    if rounded_value >= num_bins:
                        raise ValueError(
                            f"{key}={entity.params[key]} overflowed to {rounded_value}..."
                        )
                    entity.params[key] = rounded_value

                elif dtype == str:
                    entity.params[key] = normalization_values[
                        normalization_strategy
                    ].index(entity.params[key])

                elif dtype == int:
                    pass  # The value is already discretized

    def undiscretize_and_unnormalize(self, num_bins, normalization_values):
        """Reverse the normalization/discretization process.

        Args:
            num_bins: int.
            normalization_values: Dict[str, List[Union[float, str]]]. See description
                in self.normalize().
        """
        for entity in self.entities:
            for key in entity.params:

                normalization_strategy = entity.PARAMS_DEFINITION[key]["normalise"]
                dtype = entity.PARAMS_DEFINITION[key]["dtype"]

                if dtype == float:  # value is in [0, 1]
                    undiscretized = entity.params[key] / num_bins
                    min_val, max_val = normalization_values[normalization_strategy]
                    entity.params[key] = undiscretized * (max_val - min_val) + min_val

                elif dtype == str:
                    entity.params[key] = normalization_values[normalization_strategy][
                        entity.params[key]
                    ]

                elif dtype == int:
                    pass  # The value is already un-discretized/un-normalised.

    def sort_entities(self, sort_type):
        """Sort language entities.

        The instances belonging to each entity type will be sorted.
            E.g. all make_wall commands are sorted, then make_door, etc.

        Args:
            sort_type: str. MUST be in ["lex", "random"].
        """
        sorted_entities = []
        for ENTITY_CLASS in ALL_ENTITY_CLASSES:

            sort_keys = []
            entities_to_sort = []

            for entity in self.entities:
                if entity.COMMAND_STRING == ENTITY_CLASS.COMMAND_STRING:
                    sort_keys.append(entity.sort_key(sort_type))
                    entities_to_sort.append(entity)

            if entities_to_sort:  # Note: may be no windows or no bboxes
                sorted_idx = np.lexsort(np.stack(sort_keys).T)
                sorted_entities.extend([entities_to_sort[i] for i in sorted_idx])

        self.entities = sorted_entities

    def assign_doors_windows_to_walls(self):
        wall_entities = [x for x in self.entities if isinstance(x, WallEntity)]

        for entity in self.entities:
            if isinstance(entity, DoorEntity) or isinstance(entity, WindowEntity):
                num_walls = len(wall_entities)
                wall_starts = np.array(
                    [
                        [wall_ent.params["a_x"], wall_ent.params["a_y"]]
                        for wall_ent in wall_entities
                    ]
                )  # [N_walls, 2]
                wall_ends = np.array(
                    [
                        [wall_ent.params["b_x"], wall_ent.params["b_y"]]
                        for wall_ent in wall_entities
                    ]
                )  # [N_walls, 2]
                walls = [(wall_starts[i], wall_ends[i]) for i in range(num_walls)]
                # List of [(x1,y1), (x2,y2)] wall start/ends.

                pos_xy = np.array(
                    [entity.params["position_x"], entity.params["position_y"]]
                )
                parent_idx = np.argmin(
                    [point_to_line_seg_dist(pos_xy, w) for w in walls]
                )
                entity.assign_parent_wall_entity(wall_entities[parent_idx])

    def generate_language_string(self):
        """Write SceneScript Language string."""
        all_lines = []
        for entity in self.entities:
            new_line = ", ".join(
                [entity.COMMAND_STRING]
                + [f"{key}={entity.params[key]}" for key in entity.PARAMS_DEFINITION]
            )
            all_lines.append(new_line)

        return "\n".join(all_lines)

    @staticmethod
    def from_seq_value(seq_value):
        """Convert a sequence value to a language sequence.

        Args:
            seq_value: [T] torch.LongTensor.

        Returns:
            LanguageSequence.
        """
        seq = seq_value.to("cpu")
        stop_idxs = torch.where(seq == HELPER_TOKEN.STOP)[0]
        if stop_idxs.shape[0] == 0:  # No predicted STOP tokens
            return LanguageSequence([])

        # Get cut points
        stop_idx = stop_idxs[0]
        seq = seq[:stop_idx]
        part_idxs = torch.where(seq == HELPER_TOKEN.PART)[0]
        part_idxs = torch.cat([part_idxs, torch.as_tensor([stop_idx])])

        # Initialise IDs for each command
        ids_dict = {
            "make_wall": 0,
            "make_door": 1000,
            "make_window": 2000,
            "make_bbox": 3000,
        }

        entities = []
        for part_idx in range(part_idxs.shape[0] - 1):

            element_subseq = (
                seq[part_idxs[part_idx] + 1 : part_idxs[part_idx + 1]]
                - HELPER_TOKEN.NUM
            )  # remove <PART> token
            if len(element_subseq) <= 2:
                print("Warning: invalid subsequence of tokens...")
                continue

            # Get entity class
            command_value = int(element_subseq[0])
            ENTITY_CLASS = get_entity_class_from_token(command_value)
            command_str = ENTITY_CLASS.COMMAND_STRING

            # Get params
            params_subseq = element_subseq[1:]
            params = {}

            params_subseq_idx = 0
            for param_key in ENTITY_CLASS.PARAMS_DEFINITION:

                if is_id_param(param_key):  # special case
                    if param_key == "id":
                        params[param_key] = ids_dict[command_str]
                        ids_dict[command_str] += 1
                    elif param_key in ["wall0_id", "wall1_id"]:
                        pass
                    elif param_key == "bbox_id":
                        params[param_key] = ids_dict["make_bbox"] - 1  # current bbox_id
                    continue

                try:
                    params[param_key] = int(params_subseq[params_subseq_idx].item())
                except Exception as e:
                    print(
                        f"Invalid subsequence! {seq[part_idxs[part_idx] : part_idxs[part_idx + 1]]}",
                    )
                params_subseq_idx += 1

            # Append to entities
            entity = ENTITY_CLASS(params)
            entities.append(entity)

        lang_seq = LanguageSequence(entities)
        lang_seq.assign_doors_windows_to_walls()

        return lang_seq
