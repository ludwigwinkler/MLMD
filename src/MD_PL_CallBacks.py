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
from typing import Any, Dict, Optional, Union;
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import _METRIC, STEP_OUTPUT
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn, rank_zero_info
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class OverwritingModelCheckpoint(ModelCheckpoint):

	def __init__(
		self,
		**kwargs
	) -> object:

		ModelCheckpoint.__init__(
			self,
			**kwargs
		)

	def _get_metric_interpolated_filepath_name(
		self,
		monitor_candidates: Dict[str, _METRIC],
		trainer: 'pl.Trainer',
		del_filepath: Optional[str] = None,
	) -> str:
		filepath = self.format_checkpoint_name(monitor_candidates)

		# version_cnt = self.STARTING_VERSION
		# while self.file_exists(filepath, trainer) and filepath != del_filepath:
		# 	filepath = self.format_checkpoint_name(monitor_candidates, ver=version_cnt)
		# 	version_cnt += 1

		return filepath

	def format_checkpoint_name(self, metrics: Dict[str, _METRIC], ver: Optional[int] = None) -> str:
		"""Generate a filename according to the defined template.

		Example::

		    >>> tmpdir = os.path.dirname(__file__)
		    >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{epoch}')
		    >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=0)))
		    'epoch=0.ckpt'
		    >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{epoch:03d}')
		    >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=5)))
		    'epoch=005.ckpt'
		    >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{epoch}-{val_loss:.2f}')
		    >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=2, val_loss=0.123456)))
		    'epoch=2-val_loss=0.12.ckpt'
		    >>> ckpt = ModelCheckpoint(dirpath=tmpdir,
		    ... filename='epoch={epoch}-validation_loss={val_loss:.2f}',
		    ... auto_insert_metric_name=False)
		    >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=2, val_loss=0.123456)))
		    'epoch=2-validation_loss=0.12.ckpt'
		    >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{missing:d}')
		    >>> os.path.basename(ckpt.format_checkpoint_name({}))
		    'missing=0.ckpt'
		    >>> ckpt = ModelCheckpoint(filename='{step}')
		    >>> os.path.basename(ckpt.format_checkpoint_name(dict(step=0)))
		    'step=0.ckpt'

		"""

		filename = self._format_checkpoint_name(
			self.filename, metrics, auto_insert_metric_name=self.auto_insert_metric_name
		)

		# if ver is not None:
		# 	filename = self.CHECKPOINT_JOIN_CHAR.join((filename, f"v{ver}"))

		ckpt_name = f"{filename}{self.FILE_EXTENSION}"
		return os.path.join(self.dirpath, ckpt_name) if self.dirpath else ckpt_name