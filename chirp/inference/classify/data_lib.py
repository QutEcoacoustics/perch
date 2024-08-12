# coding=utf-8
# Copyright 2024 The Perch Authors.
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

"""Utilities for training separated clustering.

The core workflow consists of the following:
a) Make a collection of wav files divided by label into sub-directories.
b) Load an `interface.EmbeddingModel`.
c) Create a MergedDataset using the directory and embedding model.
  This will load all of the labeled wavs and run the embedding model over
  all of them, creating an in-memory dataset.

This dataset can then be used for all kinds of small experiments, such as
training small classifiers or evaluating clustering methods.
"""

import collections
import dataclasses
import itertools
import pandas as pd
import time
from typing import DefaultDict, Dict, List, Sequence, Tuple, Union

from chirp import audio_utils
from chirp.inference import interface
from chirp.inference import tf_examples
from etils import epath
import numpy as np
import tensorflow as tf
import tqdm

@dataclasses.dataclass
class LabelInfo:
  """Container for label information for a single example"""
  hot: np.ndarray[int]
  idx: List[int]
  name: List[str]

  @classmethod
  def from_hot(cls, hot: np.ndarray[int], labels: List[str]) -> 'LabelInfo':
    idx = list(np.where(hot == 1)[0])
    name = [labels[i] for i in idx]
    return cls(hot=hot, idx=idx, name=name)
  
  @classmethod
  def from_idx(cls, idx: List[int], labels: List[str]) -> 'LabelInfo':
    hot = np.zeros(len(labels), np.int32)
    name = [labels[i] for i in idx]
    for i in idx:
      hot[i] = 1
    return cls(hot=hot, idx=idx, name=name)


@dataclasses.dataclass
class MergedDataset:
  """In-memory dataset of labeled audio with embeddings.

  Attributes:
    data: Dictionary of embedding outputs.
    num_classes: Number of classes.
    embedding_dim: Dimension of embeddings.
    labels: Tuple with the labels for each file.
  """

  # The following are populated automatically from one of two classmethods.
  # data has the following keys and types
  # 'embeddings' (np.ndarray), 'filename' (np.ndarray of str), 'label' (list of list of int), 
  # 'label_str' (list of list of str), 'label_hot' (np.ndarray)
  # TODO: use a dataclass or typed dict for this to enforce the keys and types
  data: Dict[str, Union[np.ndarray, List[List]]]
  num_classes: int
  embedding_dim: int
  labels: Tuple[str, ...]


  @classmethod
  def from_folder_of_folders(cls, **kwargs):
    """
    Get a merged dataset from folder of folders. 

    This is primarily here for backwards compatibility.
    """
    print('Embedding from Folder of Folders...')
    kwargs['label_csv'] = None
    return cls.get_merged_dataset(**kwargs)

  @classmethod
  def from_csv(cls, **kwargs):
    print('Embedding from CSV...')
    return cls.get_merged_dataset(**kwargs)


  @classmethod
  def get_merged_dataset(
      cls,
      base_dir: str,
      embedding_model: interface.EmbeddingModel,
      time_pooling: str = 'mean',
      exclude_classes: Sequence[str] = (),
      load_audio: bool = False,
      target_sample_rate: int = -2,
      audio_file_pattern: str = None,
      embedding_config_hash: str = '',
      embedding_file_prefix: str = 'embeddings-',
      pad_type: str = 'zeros',
      cache_embeddings: bool = True,
      tf_record_shards: int = 1,
      max_workers: int = 5,
      label_csv = None
  ) -> 'MergedDataset':
    """Generating MergedDataset via folder-of-folders method.

    This method will scan for existing embeddings cached within the folder of
    folders and re-use those with a matching prefix. The prefix is expected to
    have a hash signature for matching configs.

    Args:
      base_dir: Base directory where either folder-of-folders of audio or
        tfrecord embeddings are stored.
      embedding_model: EmbeddingModel used to produce embeddings.
      time_pooling: Key for time pooling strategy.
      exclude_classes: Classes to skip.
      load_audio: Whether to load audio into memory. beware that this can cause
        problems with large datasets.
      target_sample_rate: Resample loaded audio to this sample rate. If -1,
        loads raw audio with no resampling. If -2, uses the embedding_model
        sample rate.
      audio_file_pattern: The glob pattern to use for finding audio files within
        the sub-folders.
      embedding_config_hash: String hash of the embedding config to identify an
        existing embeddings folder. This will be appended to the
        embedding_file_prefix, e.g. 'embeddings-1234'.
      embedding_file_prefix: Prefix for existing materialized embedding files.
        Embeddings with a matching hash will be re-used to avoid reprocessing,
        and embeddings with a non-matching hash will be ignored.
      pad_type: Padding strategy for short audio.
      cache_embeddings: Materialize new embeddings as TF records within the
        folder-of-folders.
      tf_record_shards: Number of files to materialize if writing embeddings to
        TF records.
      max_workers: Number of threads to use for loading audio.

    Returns:
      MergedDataset
    """

    # if label_csv is not None and audio_file_pattern is not None:
    #   raise ValueError('Only one of csv_path and audio_file_pattern should be specified.')


    st = time.time()

    existing_merged = None
    existing_embedded_srcs = []
    if embedding_config_hash:
      print('Checking for existing embeddings...')

      base_path = epath.Path(base_dir)
      embedding_folder = (
          base_path / f'{embedding_file_prefix}{embedding_config_hash}'
      )

      if embedding_folder.exists() and any(embedding_folder.iterdir()):
        existing_merged = cls.from_tfrecords(
            base_dir, embedding_folder.as_posix(), time_pooling, exclude_classes, label_csv
        )
        existing_embedded_srcs = existing_merged.data['filename']

      print(f'Found {len(existing_embedded_srcs)} existing embeddings.')

    

    # depending on whether we are using folder-of-folders or csv
    # we will get the filepaths, example_labels, and labels in different ways
    if label_csv is not None:
      file_labels, labels = file_labels_from_csv(label_csv)
      # check that the filepaths exist 
      for fp in file_labels.keys():
        if not (base_dir / fp).exists():
          raise ValueError(f'Filepath {fp} does not exist.')
      
    else:
      file_labels, labels = file_labels_from_folder_of_folders(
          base_dir = base_dir, 
          exclude_classes = exclude_classes, 
          audio_file_pattern = audio_file_pattern, 
          embedding_file_prefix = embedding_file_prefix)
    
    labels, merged = embed_dataset(
      base_dir=base_dir,
      embedding_model=embedding_model,
      time_pooling=time_pooling,
      file_labels=file_labels,
      labels=labels,
      excluded_files=existing_embedded_srcs,
      load_audio=load_audio,
      target_sample_rate=target_sample_rate,
      pad_type=pad_type,
      max_workers=max_workers
    )


    if not merged and existing_merged is None:      raise ValueError('No embeddings or raw audio files found.')

    if not merged and existing_merged is not None:
      print('\nUsing existing embeddings for all audio source files.')
      return existing_merged

    elapsed = time.time() - st
    print(f'\n...embedded dataset in {elapsed:5.2f}s...')
    data = merged
    embedding_dim = merged['embeddings'].shape[-1]

    labels = tuple(labels)
    num_classes = len(labels)
    print(f'    found {num_classes} classes.')
    
    get_class_counts(merged['label'], merged['label_str'])
    new_merged = cls(
        data=data,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        labels=labels,
    )

    if cache_embeddings:
      if not embedding_config_hash:
        raise ValueError(
            'Embedding config hash must be specified when caching embeddings.'
        )

      new_merged.write_embeddings_to_tf_records(
          base_dir,
          embedding_config_hash,
          embedding_file_prefix,
          tf_record_shards,
      )

    if existing_merged:
      return cls.from_merged_datasets([new_merged, existing_merged])

    return new_merged

  @classmethod
  def from_tfrecords(
      cls,
      base_dir: str,
      embeddings_path: str,
      time_pooling: str,
      exclude_classes: Sequence[str] = (),
      label_csv: str = None
  ) -> 'MergedDataset':
    """Generating MergedDataset via reading existing embeddings.

    Note: this assumes the embeddings were run with folder_of_folders
    with file_id_depth=1 in the embeddings export. This classmethod can/will be
    updated for allowing a few options for specifying labels.

    Args:
      base_dir: Base directory (folder of folders of original audio)
      embeddings_path: Location of the existing embeddings.
      time_pooling: Method of time pooling.
      exclude_classes: List of classes to exclude.
      csv_path: Path to a csv file containing the labels for each audio file. If None
                will assume labels are derived from folder of folders

    Returns:
      MergedDataset
    """
    labels, merged = read_embedded_dataset(
        base_dir=base_dir,
        embeddings_path=embeddings_path,
        time_pooling=time_pooling,
        exclude_classes=exclude_classes,
        label_csv=label_csv
    )
    data = merged
    embedding_dim = merged['embeddings'].shape[-1]

    labels = tuple(labels)
    num_classes = len(labels)
    print(f'    found {num_classes} classes.')

    get_class_counts(merged['label'], merged['label_str'])


    return cls(
        data=data,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        labels=labels,
    )

  @classmethod
  def from_merged_datasets(
      cls, merged_datasets: Sequence['MergedDataset']
  ) -> 'MergedDataset':
    """Generating MergedDataset from a sequence of MergedDatasets.

    This assumes that the given merged datasets are compatible, i.e. they were
    generated with the same options and embedding configurations.

    Args:
      merged_datasets: Sequence of compatible MergedDatasets.

    Returns:
      MergedDataset
    """

    embedding_dim = merged_datasets[0].embedding_dim
    num_classes = merged_datasets[0].num_classes
    labels = merged_datasets[0].labels
    data = {}

    for merged_dataset in merged_datasets[1:]:
      # TODO: Improve compatibility checking to use config hashes.
      if (
          embedding_dim != merged_dataset.embedding_dim
          or num_classes != merged_dataset.num_classes
          or labels != merged_dataset.labels
      ):
        raise ValueError('Given merged datasets are not compatible.')

    for key in merged_datasets[0].data.keys():
      data_arrays = [merged_data.data[key] for merged_data in merged_datasets]
      if type(data_arrays[0]) is list:
        data[key] = [item for sublist in data_arrays for item in sublist]
      else:
        data[key] = np.concatenate(data_arrays)

    return cls(
        data=data,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        labels=labels,
    )

  def embeddings_to_tf_examples(self) -> Sequence[tf.train.Example]:
    """Return a dictionary of embedding tf.Examples keyed by label_str."""
    examples = []
    embeddings = self.data['embeddings']
    filename = self.data['filename']

    for embedding, filename in zip(embeddings, filename):
      examples.append(
          tf_examples.model_outputs_to_tf_example(
              model_outputs=interface.InferenceOutputs(embedding),
              file_id=filename,
              audio=np.empty(1),
              timestamp_offset_s=0,
              write_embeddings=True,
              write_logits=False,
              write_separated_audio=False,
              write_raw_audio=False,
          )
      )

    return examples

  def write_embeddings_to_tf_records(
      self,
      base_dir: str,
      embedding_config_hash: str,
      embedding_file_prefix: str = 'embeddings-',
      tf_record_shards: int = 1,
  ) -> None:
    """Materialize MergedDataset embeddings as TF records to folder-of-folders.

    Args:
      base_dir: Base directory where either folder-of-folders of audio or
        tfrecord embeddings are stored.
      embedding_config_hash: String hash of the embedding config to identify an
        existing embeddings folder. This will be appended to the
        embedding_file_prefix, e.g. 'embeddings-1234'.
      embedding_file_prefix: Prefix for existing materialized embedding files.
        Embeddings with a matching hash will be re-used to avoid reprocessing,
        and embeddings with a non-matching hash will be ignored.
      tf_record_shards: Number of files to materialize if writing embeddings to
        TF records.
    """
    embedding_examples = self.embeddings_to_tf_examples()
    output_dir = (
        epath.Path(base_dir) / f'{embedding_file_prefix}{embedding_config_hash}'
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    with tf_examples.EmbeddingsTFRecordMultiWriter(
        output_dir=output_dir.as_posix(), num_files=tf_record_shards
    ) as file_writer:
      for example in embedding_examples:
        file_writer.write(example.SerializeToString())
      file_writer.flush()

  def create_random_train_test_split(
      self,
      train_ratio: float | None,
      train_examples_per_class: int | None,
      seed: int,
      exclude_classes: Sequence[int] = (),
      exclude_eval_classes: Sequence[int] = (),
  ):
    """
    Generate a train/test split with a target number of train examples.

    Args:
      train_ratio: Ratio of examples to use for training.
      train_examples_per_class: Number of examples to use for training.
      seed: Random seed for splitting.
      exclude_classes: Classes to exclude from training.
      exclude_eval_classes: Classes to exclude from evaluation.

    Returns:
      Tuple of train_locs, test_locs, class_locs.

      Class locs is a mapping labels to the indexes that contain that label. This may
      Sum to more than the number of examples if there are multiple labels per example.

    In the case that examples have more than one label, this might result in
    not enough test examples for some classes depending on the nature of the dataset. 
    This is because we first calculate the number of training examples needed for each class.
    Then for each example we add it to the training splt if any of its classes have not reached the
    target number of training examples, otherwise it goes in the test split. If it just happens that
    a particular class reaches its target number of training examples, and then further examples for that
    class also belong to another class that has not reached its training target, then those examples
    will also go in the training set. 
    
    """
    if train_ratio is None and train_examples_per_class is None:
      raise ValueError(
          'Must specify one of train_ratio and examples_per_class.'
      )
    elif train_ratio is not None and train_examples_per_class is not None:
      raise ValueError(
          'Must specify only one of train_ratio and examples_per_class.'
      )

    # Use a seeded shuffle to get a random ordering of the data.
    locs = list(range(len(self.data['label'])))
    np.random.seed(seed)
    np.random.shuffle(locs)

    # flatten labels only if self.data['label'] is a list of lists
    flattened_labels = [l for ex in self.data['label'] for l in ex]

    classes = set(flattened_labels)
    class_counts = {cl: flattened_labels.count(cl) for cl in classes}
    if train_examples_per_class is not None:
      class_limits = {cl: train_examples_per_class for cl in classes}
    else:
      class_limits = {cl: train_ratio * class_counts[cl] for cl in classes}

    
    class_locs = {cl: [] for cl in classes}
    train_locs = []
    test_locs = []
    for loc in locs:
      item_labels = self.data['label'][loc]
      # we exclude an item if any of its labels are in exclude_classes
      if any(exclude_item in item_labels for exclude_item in exclude_classes):
        continue

      for cl in item_labels:
        use_for_train = False
        if len(class_locs[cl]) < class_limits[cl]:
          class_locs[cl].append(loc)
          use_for_train = True
      if use_for_train:
        train_locs.append(loc)
      elif not any(exclude_item in item_labels for exclude_item in exclude_eval_classes):
        test_locs.append(loc)
    train_locs = np.array(train_locs)
    test_locs = np.array(test_locs)
    return train_locs, test_locs, class_locs


def pool_time_axis(embeddings, pool_method, axis=1):
  """Apply pooling over the specified axis."""
  if pool_method == 'mean':
    if embeddings.shape[axis] == 0:
      return embeddings.sum(axis=axis)
    return embeddings.mean(axis=axis)
  elif pool_method == 'max':
    return embeddings.max(axis=axis)
  elif pool_method == 'mid':
    t = embeddings.shape[axis] // 2
    return embeddings[:, t]
  elif pool_method == 'flatten':
    if len(embeddings.shape) != 3 and axis != 1:
      raise ValueError(
          'Can only flatten time for embeddings with shape [B, T, D].'
      )
    depth = embeddings.shape[-1]
    time_steps = embeddings.shape[1]
    return embeddings.reshape([embeddings.shape[0], time_steps * depth])
  raise ValueError('Unrecognized reduction method.')


def _pad_audio(
    audio: np.ndarray, target_length: int, pad_type: str = 'zeros'
) -> np.ndarray:
  """Pad audio to target_length."""
  if len(audio.shape) > 1:
    raise ValueError('audio should be a flat array.')
  if audio.shape[0] >= target_length:
    return audio
  if pad_type == 'zeros':
    pad_amount = target_length - audio.shape[0]
    front = pad_amount // 2
    back = pad_amount - front
    return np.pad(audio, [(front, back)], 'constant')
  elif pad_type == 'repeat':
    # repeat audio until longer than target_length.
    num_repeats = target_length // audio.shape[0] + 1
    repeated_audio = np.repeat(audio, num_repeats, axis=0)
    start = repeated_audio.shape[0] - target_length // 2
    padded = repeated_audio[start : start + target_length]
    return padded
  raise ValueError('Unrecognized padding method.')


# TODO: add alternative labeling strategies as options
def labels_from_folder_of_folders(
    base_dir: str,
    exclude_classes: Sequence[str] = (),
    embedding_file_prefix: str = 'embeddings-',
) -> Sequence[str]:
  """Returns the labels from the given folder of folders.

  Args:
    base_dir: Folder of folders directory containing audio or embedded data.
    exclude_classes: Classes to skip.
    embedding_file_prefix: Folders containing existing embeddings that will be
      ignored when determining labels.
  """
  base_dir = epath.Path(base_dir)
  sub_dirs = sorted([p.name for p in base_dir.glob('*') if p.is_dir()])
  if not sub_dirs:
    raise ValueError(
        'No subfolders found in base directory. Audio will be '
        'matched as "base_dir/*/*.wav", with the subfolders '
        'indicating class names.'
    )

  labels = []
  for d in sub_dirs:
    if d in exclude_classes:
      continue
    if d.startswith(embedding_file_prefix):
      continue
    labels.append(d)

  return labels

def write_csv_from_folder_of_folders(
  base_dir: str,
  exclude_classes: Sequence[str] = (),
  audio_file_pattern: str = '*.wav',
  embedding_file_prefix: str = 'embeddings-',
  labels_csv: str = 'labels.csv',
  merge_matching_basename: bool = True
):
  """
  Writes a csv of filepaths and labels from a folder of folders

  Args:
    base_dir: str; the base directory containing the folder of folders
    exclude_classes: Sequence[str]; a list of classes to exclude
    audio_file_pattern: str; a glob pattern to use for finding audio files within the sub-folders
    embedding_file_prefix: str; the prefix for existing materialized embedding files. 
                                Embeddings with a matching hash will be re-used to avoid reprocessing,
                                and embeddings with a non-matching hash will be ignored.
    labels_csv: str; the name of the csv file to write the labels to
    merge_matching_basename: bool; whether to merge labels if the basename is the same. 
                                   If folder of folder has been used and examples containing more than one label have been
                                   duplicated to more than one folder, this will merge the labels into one row, referencing only 
                                   one copy of the example files. 

  If there is a folder of folders where each folder name is a label, but we want to 
  change to using a csv to associate labels with files (maybe because we want to 
  allow multiple labels per example), this function will make this conversion.
  """

  example_labels, labels = file_labels_from_folder_of_folders(
    base_dir=base_dir, 
    exclude_classes=exclude_classes, 
    audio_file_pattern=audio_file_pattern, 
    embedding_file_prefix=embedding_file_prefix)
  
  labels_hot_list = [ex.hot for ex in example_labels.values()]
  # convert to a dict where there is a key:value for each label, with the key being the label name and the value being a list of
  # whether the example has that label or not
  data_dict = {'filepath': example_labels.keys(), **{label: [labels_hot_list[i][j] for i in range(len(example_labels))] for j, label in enumerate(labels)}}
  df = pd.DataFrame(data_dict)

  if merge_matching_basename:

    # merge labels if the basename is the same. Multiple labels per example may have been captured by duplicating the 
    # example file into more than one subfolder. In this case, we keep the first file and add the labels from the other matching files
    # to that row. 

    # temp basename column for grouping, which is dropped later
    df['basename'] = df['filepath'].apply(lambda x: epath.Path(x).name)
    df = df.groupby('basename').agg({**{'filepath': 'first'}, **{col: 'sum' for col in labels}}).reset_index(drop=True)
    df.drop(columns=['basename'], inplace=True, errors='ignore')


  csv_path = epath.Path(base_dir) / epath.Path(labels_csv)

  df.to_csv(csv_path, index=False)


def labels_from_filelist(filelist: Sequence[str], labels = None) -> Tuple[Dict[epath.Path, LabelInfo], Sequence[str]]:
  """
  Returns the labels from a list of filepaths, assuming that the first part of the path is the label.

  Args:
    filelist: Sequence[str]; a list of filepaths
    labels: Sequence[str]; a list of labels to use. If None, the labels will be inferred from the filepaths

  Returns:
    a dictionary of filepaths to labels
    labels: Sequence[str]; a list of labels.
  
  This is relevant only to single-label classification based on folder-of-folders. 
  The labels argument is so that the label indexes and one-hot encoding can include classes that are not
  present in the filelist.
  """

  example_labels_str = [epath.Path(fp).parts[0] for fp in filelist]

  if labels is None:
    labels = set(example_labels_str)
  label_dict = get_label_dict(labels)

  # beware that label dicts are references to the same object for the same label
  example_labels = [label_dict[label] for label in example_labels_str]

  example_labels = dict(zip(filelist, example_labels))

  return example_labels, labels
   
   
  

def file_labels_from_folder_of_folders(
    base_dir: str, 
    exclude_classes: Sequence[str] = (), 
    audio_file_pattern: str = '*',
    embedding_file_prefix: str = 'embeddings-',
    ) -> Tuple[Dict[epath.Path, LabelInfo], Sequence[str]]:
  """
  Gets the filepaths and associated labels based on a folder of folders

  Args:
    base_dir: str;  the base directory containing the folder of folders
    exclude_classes: Sequence[str]; a list of classes to exclude
    audio_file_pattern: str; a glob pattern to use for finding audio files within the sub-folders
    embedding_file_prefix: str;  the prefix for existing materialized embedding files. 
                                Embeddings with a matching hash will be re-used to avoid reprocessing,
                                and embeddings with a non-matching hash will be ignored.

  Returns:
    - a mapping of filepaths to labels
    - a list of labels
  
  The filepath will be relative to the base directory, and therefore the first part of the
  path with be a folder name that is the label for the example. 
  The embedding_file_prefix is only there so that the embeddings folder is not included in the labels.
  
  """
  
  print('Checking for new sources to embed from Folder of Folders...')

  base_dir = epath.Path(base_dir)
  
  all_filepaths = []

  labels = labels_from_folder_of_folders(base_dir, exclude_classes, embedding_file_prefix)
  for label in labels:

    # Get filepaths excluding embedding files
    filepaths = [
        fp.relative_to(base_dir)
        for fp in (base_dir / label).glob(audio_file_pattern)
        if not fp.name.startswith(embedding_file_prefix)
    ]

    if not filepaths:
      raise ValueError(
          'No files matching {} were found in directory {}'.format(
              audio_file_pattern, base_dir / label
          )
      )
    
    all_filepaths = all_filepaths + filepaths
  
  file_labels, labels = labels_from_filelist(all_filepaths, labels)

  return file_labels, labels


def file_labels_from_csv(labels_csv: str) -> Tuple[Dict[epath.Path, LabelInfo], Sequence[str]]:
  """
  Creates a dict mapping filepaths to their labels, and a list of labels

  Args:
    labels_csv: the path to the csv file containing the labels

  Returns:
    a list of filepaths mapped to labels
    a list of labels

  """

  print('Checking for new sources to embed from csv ...')

  # Read the labels csv as a pandas dataframe
  labels_df = pd.read_csv(labels_csv)

  # labels_df has a column 'filename' and a column for each label. 
  # The column names are the label names.
  # the 'filename' column contains the relative path to the audio file.
  # the label columns contain a 1 if the audio file belongs to that or 0 otherwise.

  # get a list of labels from the column names, making sure
  # not to include the 'filename' column
  labels = [col for col in labels_df.columns if col != 'filepath']

  # check that labels is in alphabetical order
  # to avoid mixups we keep the labels in alphabetical order, 
  # and we also want to keep consistency between the labels in the csv
  # and how we store them in the embeddings
  if labels != sorted(labels):
    raise ValueError('Label columns in csv must be in alphabetical order.')

  example_labels = [
    LabelInfo.from_hot(np.array(row[labels], np.int32), labels) 
    for idx, row in labels_df.iterrows()
  ]

  filepaths = [epath.Path(row['filepath']) for idx, row in labels_df.iterrows()]
  example_labels = dict(zip(filepaths, example_labels))

  return example_labels, labels


def embed_dataset(
    base_dir: str,
    embedding_model: interface.EmbeddingModel,
    time_pooling: str,
    file_labels: list,
    labels: list,
    excluded_files: Sequence[str] = (),
    load_audio: bool = False,
    target_sample_rate: int = -1,
    pad_type: str = 'zeros',
    max_workers: int = 5,
) -> Tuple[Sequence[str], Dict[str, np.ndarray]]:
  """Add embeddings to an eval dataset.

  Embed a dataset, creating an in-memory copy of all data with embeddings added.
  The base_dir should contain folders corresponding to classes, and each
  sub-folder should contina audio files for the respective class.

  Note that any audio files in the base_dir directly will be ignored.

  Args:
    base_dir: Directory contianing audio data.
    embedding_model: Model for computing audio embeddings.
    time_pooling: Key for time pooling strategy.
    labels_csv: CSV file containing the labels for each audio file.
    load_audio: Whether to load audio into memory.
    target_sample_rate: Resample loaded audio to this sample rate. If -1, loads
      raw audio with no resampling. If -2, uses the embedding_model sample rate.
    pad_type: Padding style for short audio.
    max_workers: Number of threads to use for loading audio.

  Returns:
    Ordered labels and a Dict contianing the entire embedded dataset.
  """

  file_labels = {
      fp.as_posix(): v 
      for fp, v in file_labels.items() 
      if fp.as_posix() not in excluded_files
  }

  print(f'Embedding {len(file_labels)} files...')
  
  base_dir = epath.Path(base_dir)

  if hasattr(embedding_model, 'window_size_s'):
    window_size = int(
        embedding_model.window_size_s * embedding_model.sample_rate
    )
  else:
    window_size = -1

  if target_sample_rate == -2:
    target_sample_rate = embedding_model.sample_rate

  merged = collections.defaultdict(list)

  audio_loader = lambda fp, offset: _pad_audio(
      np.asarray(audio_utils.load_audio(fp, target_sample_rate)),
      window_size,
      pad_type,
  )

  full_file_paths = [base_dir / epath.Path(fp) for fp in file_labels.keys()]
  audio_iterator = audio_utils.multi_load_audio_window(
      audio_loader=audio_loader,
      filepaths=full_file_paths,
      offsets=None,
      max_workers=max_workers,
      buffer_size=64,
  )

  filepaths = list(file_labels.keys())
  example_labels = list(file_labels.values())

  for idx, audio in enumerate(tqdm.tqdm(audio_iterator)):
    outputs = embedding_model.embed(audio)
    if outputs.embeddings is None:
      raise ValueError('Embedding model did not produce any embeddings!')
    # If the audio was separated then the raw audio is in the first channel.
    # Embedding shape is either [B, F, C, D] or [F, C, D] so channel is
    # always -2.
    channel_pooling = (
        'squeeze' if outputs.embeddings.shape[-2] == 1 else 'first'
    )
    embeds = outputs.pooled_embeddings(time_pooling, channel_pooling)
    merged['embeddings'].append(embeds)
    if load_audio:
      merged['audio'].append(audio)

    merged['filename'].append(filepaths[idx])
    merged['label'].append(example_labels[idx].idx)
    merged['label_str'].append(example_labels[idx].name)
    merged['label_hot'].append(example_labels[idx].hot)

  outputs = {}


  for k in merged.keys():
    if k in ['label_str', 'label']:
      # we don't want to np.stack label_str or label because 
      # they are variable length depending on how many labels are present in each example
      outputs[k] = merged[k]
    else:
      outputs[k] = np.stack(merged[k])

  return labels, outputs



def read_embedded_dataset(
    base_dir: str,
    embeddings_path: str,
    time_pooling: str,
    exclude_classes: Sequence[str] = (),
    tensor_dtype: str = 'float32',
    label_csv: str = None
):
  """Read pre-saved embeddings to memory from storage.

  This function reads a set of embeddings that has already been generated
  to load as a MergedDataset via from_tfrecords(). The embeddings could be saved
  in one folder or be contained in multiple subdirectories. This function
  produces the same output as embed_dataset(), except (for now) we don't allow
  for the optional loading of the audio (.wav files). However, for labeled data,
  we still require the base directory containing the folder-of-folders with the
  audio data to produce the labels, or a csv providing the labels. 
  If there are no subfolders or csv, no labels will be created.

  Args:
    base_dir: Base directory where audio may be stored in a subdirectories,
      where the folder represents the label
    embeddings_path: Location of the existing embeddings as TFRecordDataset.
    time_pooling: Method of time pooling.
    exclude_classes: List of classes to exclude.
    tensor_dtype: Tensor dtype used in the embeddings tfrecords.
    label_csv: Path to a csv file containing the labels for each audio file. If None
               will assume labels are derived from folder of folders

  Returns:
    Ordered labels and a Dict contianing the entire embedded dataset.
  """

  output_dir = epath.Path(embeddings_path)
  fns = [fn for fn in output_dir.glob('embeddings-*')]
  ds = tf.data.TFRecordDataset(fns)
  parser = tf_examples.get_example_parser(tensor_dtype=tensor_dtype)
  ds = ds.map(parser)

  # if we are using folder-of-folders for labels the set of all labels comes from
  # the folder names in base_dir, and the label for each example comes from the 
  # first part of the path stored with the tf record
  # If we are using the csv for labels, the set of all labels and the labels for each
  # example comes from the csv file
  if label_csv is None:
    labels = labels_from_folder_of_folders(base_dir, exclude_classes)
    label_dict = get_label_dict(labels)
    # label name is the top level folder in the path
    get_label = lambda ex: label_dict[ex['filename'].decode().split('/')[0]]
  else:
    file_labels, labels = file_labels_from_csv(label_csv)
    # TODO: this will key error if the previously embedded example no longer exists in 
    # the csv. We need to handle this
    get_label = lambda ex: file_labels[epath.Path(ex['filename'].decode())]

  merged = collections.defaultdict(list)
  
  for ex in ds.as_numpy_iterator():
    # Embedding has already been pooled into single dim.
    if len(ex['embedding'].shape) == 1:
      outputs = ex['embedding']
    else:
      outputs = interface.pool_axis(ex['embedding'], -3, time_pooling)
      if ex['embedding'].shape[-2] == 1:
        channel_pooling = 'squeeze'
      else:
        channel_pooling = 'first'
      outputs = interface.pool_axis(outputs, -2, channel_pooling)

    merged['embeddings'].append(outputs)
    merged['filename'].append(ex['filename'].decode())
    merged['label'].append(get_label(ex).idx)
    merged['label_str'].append(get_label(ex).name)
    merged['label_hot'].append(get_label(ex).hot)

  outputs = {}
  for k in merged.keys():
    if k in ['label_str', 'label']:
      # we don't want to np.stack label_str or label because 
      # they are variable length depending on how many labels are present in each example
      outputs[k] = merged[k]
    else:
      outputs[k] = np.stack(merged[k])
  
  return labels, outputs



def get_label_dict(labels) -> DefaultDict[str, LabelInfo]:
  """
  Creates a set of labels for single label classification
  idx and str are lists of length one to allow compatibility with multilabel datasets
  """

  label_dict = collections.defaultdict(dict)
  for label_idx, label in enumerate(labels):
    label_dict[label] = LabelInfo.from_idx([label_idx], labels)

  return label_dict


def get_class_counts(
    labels_idx: list[list[int]], 
    labels_str: list[list[str]]) -> DefaultDict[Tuple[int, str], int]:
  """
  Gets the number of examples per class as a defaultdict
  where the key is a tuple of label index and label string
  and the value is the count of examples in that class. 

  Args:
    label: list of lists label indexes for each example
    label_str: list of lists of label strings for each example

  Returns:
    The number of the number of examples per class

  The total of the class counts may be more than the number of examples
  because an example may have multiple labels
  """
    
  class_counts = collections.defaultdict(int)
  for cl_list, cl_str_list in zip(labels_idx, labels_str):
    for cl, cl_str in zip(cl_list, cl_str_list):
      class_counts[(cl, cl_str)] += 1
  for (cl, cl_str), count in sorted(class_counts.items()):
    print(f'    class {cl_str} / {cl} : {count}')

  return class_counts

