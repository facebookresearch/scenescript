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

from abc import ABC, abstractmethod

import numpy as np


class BaseEntity(ABC):

    @property
    @abstractmethod
    def COMMAND_STRING(self):
        pass

    @property
    @abstractmethod
    def PARAMS_DEFINITION(self):
        pass

    @property
    @abstractmethod
    def TOKEN(self):
        pass

    @abstractmethod
    def extent(self):
        pass

    @abstractmethod
    def rotate(self, rotation_angle):
        pass

    @abstractmethod
    def translate(self, translation):
        pass

    @abstractmethod
    def lex_sort_key(self):
        pass

    def random_sort_key(self):
        """Compute sorting key for random sorting.

        Returns:
            a [1] np.ndarray.
        """
        return np.random.rand(1)  # [1]

    def sort_key(self, sort_type):
        """Compute sorting key.

        Args:
            sort_type: str.

        Returns:
            an np.ndarray.
        """
        assert sort_type in ["lex", "random"]
        if sort_type == "lex":
            return self.lex_sort_key()
        elif sort_type == "random":
            return self.random_sort_key()
