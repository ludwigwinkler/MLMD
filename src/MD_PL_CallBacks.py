# Copyright The PyTorch Lightning team.
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

"""
Model Checkpointing
===================

Automatically save model checkpoints during training.

"""

import os
import re
import yaml
from copy import deepcopy
from typing import Any, Dict, Optional, Union
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn, rank_zero_info
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class CustomModelCheckpoint(ModelCheckpoint):

	def __init__(
		self,
		filepath: Optional[str] = None,
		monitor: Optional[str] = None,
		verbose: bool = False,
		save_last: Optional[bool] = None,
		save_top_k: Optional[int] = None,
		save_weights_only: bool = False,
		mode: str = "auto",
		period: int = 1,
		prefix: str = "",
	):

		ModelCheckpoint.__init__(
			self,
			filepath = filepath,
			monitor = monitor,
			verbose = verbose,
			save_last = save_last,
			save_top_k= save_top_k,
			save_weights_only = save_weights_only,
			mode= mode,
			period= period,
			prefix= prefix
		)

	def _get_metric_interpolated_filepath_name(self, epoch, ckpt_name_metrics):
		'''Overwrites the versioning system'''
		filepath = self.format_checkpoint_name(epoch, ckpt_name_metrics)
		# version_cnt = 0
		# while self._fs.exists(filepath):
		# 	filepath = self.format_checkpoint_name(
		# 		epoch, ckpt_name_metrics, ver=version_cnt
		# 	)
		# 	# this epoch called before
		# 	version_cnt += 1
		return filepath
