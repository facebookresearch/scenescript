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
