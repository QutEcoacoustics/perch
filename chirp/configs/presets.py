# coding=utf-8
# Copyright 2022 The Chirp Authors.
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

"""Preset Configurations.

Philosophy:
* The base_config contains values which will be re-used throughout the main
  configuration.
* Refer to values in the base_config using `get_ref` to ensure that the values
  update correctly when performing hyperparameter sweeps or testing.
* The config_utils.parse_config resolves all references, so that downstream
  code doesn't panic.
"""

from chirp.config_utils import callable_config as _c
from ml_collections import config_dict


def get_base_config(**kwargs):
  """Create the base config object.

  Contains common values and FieldReferences.

  Args:
    **kwargs: Values to add or override in the base config.

  Returns:
    Config dict containing common default values.
  """
  config = config_dict.ConfigDict()
  config.sample_rate_hz = 32000
  config.train_window_size_s = 5
  config.eval_window_size_s = 5
  config.frame_rate_hz = 100
  config.num_channels = 160
  config.batch_size = 64
  config.add_taxonomic_labels = True
  config.target_class_list = 'xenocanto'
  config.num_train_steps = 1_000_000
  config.pad_mask = False
  config.tfds_data_dir = ''
  config.update(kwargs)
  return config


def get_base_init_config(config: config_dict.ConfigDict,
                         **kwargs) -> config_dict.ConfigDict:
  """Default init config."""
  init_config = config_dict.ConfigDict()
  init_config.input_shape = (config.get_ref('train_window_size_s') *
                             config.get_ref('sample_rate_hz'),)
  init_config.learning_rate = 0.0001
  init_config.rng_seed = 0
  init_config.target_class_list = config.get_ref('target_class_list')
  init_config.update(**kwargs)
  return init_config


def get_base_train_config(config: config_dict.ConfigDict,
                          **kwargs) -> config_dict.ConfigDict:
  train_config = config_dict.ConfigDict()
  train_config.num_train_steps = config.get_ref('num_train_steps')
  train_config.log_every_steps = 250
  train_config.checkpoint_every_steps = 25_000
  train_config.update(**kwargs)
  return train_config


def get_base_eval_config(config: config_dict.ConfigDict,
                         **kwargs) -> config_dict.ConfigDict:
  eval_config = config_dict.ConfigDict()
  eval_config.num_train_steps = config.get_ref('num_train_steps')
  eval_config.eval_steps_per_checkpoint = 1000
  eval_config.tflite_export = True
  eval_config.update(**kwargs)
  return eval_config


def get_pcen_melspec_config(
    config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  frontend_stride = (
      config.get_ref('sample_rate_hz') // config.get_ref('frame_rate_hz'))
  return _c(
      'frontend.MelSpectrogram',
      features=config.get_ref('num_channels'),
      stride=frontend_stride,
      kernel_size=2_048,  # ~0.08 * 32,000
      sample_rate=config.get_ref('sample_rate_hz'),
      freq_range=(60, 10_000),
      scaling_config=_c('frontend.PCENScalingConfig', conv_width=256))


def get_supervised_train_pipeline(
    config: config_dict.ConfigDict, mixin_prob: float,
    train_dataset_dir: str) -> config_dict.ConfigDict:
  """Create the supervised training data pipeline."""
  train_dataset_config = config_dict.ConfigDict()
  train_dataset_config.pipeline = _c(
      'pipeline.Pipeline',
      ops=[
          _c('pipeline.Shuffle', shuffle_buffer_size=512),
          _c('pipeline.OnlyJaxTypes'),
          _c('pipeline.ConvertBirdTaxonomyLabels',
             source_namespace='ebird2021',
             target_class_list=config.get_ref('target_class_list'),
             add_taxonomic_labels=config.get_ref('add_taxonomic_labels')),
          _c('pipeline.MixAudio', mixin_prob=mixin_prob),
          _c('pipeline.Pad',
             pad_size=config.get_ref('train_window_size_s'),
             add_mask=config.get_ref('pad_mask')),
          _c('pipeline.RandomSlice',
             window_size=config.get_ref('train_window_size_s')),
          _c('pipeline.Batch',
             batch_size=config.get_ref('batch_size'),
             split_across_devices=True),
          _c('pipeline.RandomNormalizeAudio', min_gain=0.15, max_gain=0.25),
          _c('pipeline.Repeat')
      ])
  train_dataset_config.split = 'train'
  train_dataset_config.tfds_data_dir = config.get_ref('tfds_data_dir')
  train_dataset_config.dataset_directory = train_dataset_dir
  return train_dataset_config


def get_supervised_eval_pipeline(
    config: config_dict.ConfigDict,
    eval_dataset_dir: str) -> config_dict.ConfigDict:
  """Create Caples eval data pipeline."""
  eval_dataset_config = config_dict.ConfigDict()
  eval_dataset_config.pipeline = _c(
      'pipeline.Pipeline',
      ops=[
          _c('pipeline.OnlyJaxTypes'),
          _c('pipeline.ConvertBirdTaxonomyLabels',
             source_namespace='ebird2021',
             target_class_list=config.get_ref('target_class_list'),
             add_taxonomic_labels=config.get_ref('add_taxonomic_labels')),
          _c('pipeline.Pad',
             pad_size=config.get_ref('eval_window_size_s'),
             random=False,
             add_mask=config.get_ref('pad_mask')),
          _c('pipeline.Slice',
             window_size=config.get_ref('eval_window_size_s'),
             start=0.0),
          _c('pipeline.Batch',
             batch_size=config.get_ref('batch_size'),
             split_across_devices=True),
          _c('pipeline.NormalizeAudio', target_gain=0.2),
      ])
  eval_dataset_config.split = 'train'
  eval_dataset_config.tfds_data_dir = config.get_ref('tfds_data_dir')
  eval_dataset_config.dataset_directory = eval_dataset_dir
  return eval_dataset_config
