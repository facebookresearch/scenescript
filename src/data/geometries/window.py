# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .door import DoorEntity


class WindowEntity(DoorEntity):

    COMMAND_STRING = "make_window"

    TOKEN = 3
