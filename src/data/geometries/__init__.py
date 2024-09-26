# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .bbox import BboxEntity
from .door import DoorEntity
from .wall import WallEntity
from .window import WindowEntity

ALL_ENTITY_CLASSES = [
    WallEntity,
    DoorEntity,
    WindowEntity,
    BboxEntity,
]


def get_entity_class_from_token(command_value: int):
    """Get the entity class from the integer token."""
    for entity_class in ALL_ENTITY_CLASSES:
        if entity_class.TOKEN == command_value:
            return entity_class
    raise ValueError(f"Unknown command token: {command_value}")


def get_entity_class_from_string(command_string: str):
    """Get the entity class from the integer token."""
    for entity_class in ALL_ENTITY_CLASSES:
        if entity_class.COMMAND_STRING == command_string:
            return entity_class
    raise ValueError(f"Unknown command token: {command_string}")
