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

import torch
import yaml
from omegaconf import OmegaConf

from src.data.geometries import DoorEntity, WindowEntity
from src.data.language_sequence import LanguageSequence
from src.data.point_cloud import PointCloud
from src.networks.scenescript_model import SceneScriptWrapper


def test_point_cloud_loading():

    global_points_path = (
        "/home/chrisdxie/local_data/chris_apartment_semidense_points.csv.gz"
    )
    my_point_cloud = PointCloud.load_from_file(global_points_path)
    extent = my_point_cloud.extent()
    print("Extent:")
    print(extent)
    print()

    world_shift = torch.tensor(
        [
            (extent["min_x"] + extent["max_x"]) / 2,
            (extent["min_y"] + extent["max_y"]) / 2,
            (extent["min_z"] + extent["max_z"]) / 2,
        ]
    ).float()
    my_point_cloud.translate(-world_shift)
    print("Translated extent:")
    print(my_point_cloud.extent())
    print()

    normalization_values = {"world": [-16.0, 16.0]}
    my_point_cloud.normalize_and_discretize(640, normalization_values)
    print("Normalized/Discretized extent:")
    print(my_point_cloud.extent())
    print()
    print(my_point_cloud.coords.shape, my_point_cloud.coords[:5])


def test_language_loading():
    language_path = "/home/chrisdxie/local_data/scenescript_language_example.txt"
    # Adapted from here: https://www.internalfb.com/manifold/explorer/euston/tree/dump/language_pred/global_points_panoptic/pred.txt

    my_language_sequence = LanguageSequence.load_from_file(language_path)
    extent = my_language_sequence.extent()
    print("Extent:")
    print(extent)
    print()
    print(my_language_sequence.generate_language_string())
    print()

    world_shift = torch.tensor(
        [extent["min_x"], extent["min_y"], extent["min_z"]]
    ).float()
    my_language_sequence.translate(-world_shift)
    print("Translated:")
    print(my_language_sequence.generate_language_string())
    print()

    my_language_sequence.normalize_and_discretize(
        640,
        # normalization_values
        {
            "world": [0.0, 32.0],
            "width": [0.0, 25.6],
            "height": [0.0, 25.6],
            "scale": [0.0, 20.0],
            "angle": [-6.2832, 6.2832],
        },
    )
    # Note: .extent() doesn't make sense here anymore
    print("Normalized/Discretized:")
    print(my_language_sequence.generate_language_string())
    print()

    my_language_sequence.undiscretize_and_unnormalize(
        640,
        # normalization_values
        {
            "world": [0.0, 32.0],
            "width": [0.0, 25.6],
            "height": [0.0, 25.6],
            "scale": [0.0, 20.0],
            "angle": [-6.2832, 6.2832],
        },
    )
    print("Undiscretized/Unnormalized:")
    print(my_language_sequence.generate_language_string())
    print()

    my_language_sequence.sort_entities("lex")
    print("Sorted:")
    print(my_language_sequence.generate_language_string())
    print()

    for entity in my_language_sequence.entities:
        if isinstance(entity, DoorEntity) or isinstance(entity, WindowEntity):
            entity.parent_wall_entity = None
            entity.params["wall0_id"] = -1
    print("Removed wall0_id:")
    print(my_language_sequence.generate_language_string())
    print()

    my_language_sequence.assign_doors_windows_to_walls()
    print("Assigned doors/windows to walls:")
    print(my_language_sequence.generate_language_string())
    print()


def test_model_loading():

    ckpt_path = "/home/chrisdxie/local_data/test_model_ase_v1.ckpt"
    config_path = "/home/chrisdxie/fbsource/fbcode/surreal/euston/scenescript_public_internal/config/ase.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)
    model_wrapper = SceneScriptWrapper.load_from_checkpoint(ckpt_path, cfg)

    ckpt_path = "/home/chrisdxie/local_data/test_model_ase_v2.ckpt"
    config_path = "/home/chrisdxie/fbsource/fbcode/surreal/euston/scenescript_public_internal/config/model_2.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)
    model_wrapper = SceneScriptWrapper.load_from_checkpoint(ckpt_path, cfg)


def test_model_inference():

    ckpt_path = "/home/chrisdxie/local_data/test_model_ase_v2.ckpt"
    config_path = "/home/chrisdxie/fbsource/fbcode/surreal/euston/scenescript_public_internal/config/model_2.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)
    model_wrapper = SceneScriptWrapper.load_from_checkpoint(ckpt_path, cfg).cuda()

    global_points_path = (
        "/home/chrisdxie/local_data/chris_apartment_semidense_points.csv.gz"
    )
    my_point_cloud = PointCloud.load_from_file(global_points_path)

    lang_seq = model_wrapper.run_inference(
        my_point_cloud.points,
        nucleus_sampling_thresh=0.0,
        verbose=False,
    )
    language_string = lang_seq.generate_language_string()
    # print(language_string)
    with open("/home/chrisdxie/local_data/prediction.txt", "w") as f:
        print(language_string, file=f)  # Python 3.x


if __name__ == "__main__":
    # test_point_cloud_loading()
    # test_language_loading()
    # test_model_loading()
    test_model_inference()
