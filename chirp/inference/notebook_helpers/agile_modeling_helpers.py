import collections
from  dataclasses import dataclass, field
from etils.epath import Path
from ml_collections import config_dict
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy
import tensorflow as tf
import tqdm
from typing import List
from chirp.inference import colab_utils
colab_utils.initialize(use_tf_gpu=True, disable_warnings=True)

from chirp import audio_utils
from chirp.inference import baw_utils
from chirp.inference import interface
from chirp.inference import tf_examples
from chirp.inference import models
from chirp.models import metrics
from chirp.taxonomy import namespace
from chirp.inference.search import bootstrap
from chirp.inference.search import search
from chirp.inference.search import display
from chirp.inference.classify import classify
from chirp.inference.classify import data_lib


@dataclass
class AgileModelingConfig:
    data_source: str = 'ecosounds'
    baw_auth_token: str = None
    a2o_auth_token: str = None # only for backwards compatibility
    baw_domain: str = 'api.ecosounds.org'
    model_choice: str = 'perch'
    working_dir: Path = None
    labeled_data_path: Path = None
    custom_classifier_path: Path = None
    embeddings_path: Path = None 
    separation_model_key: str = ''
    separation_model_path: str = ''
    label_csv: str = None

    def __post_init__(self):
      # fields derived from other fields must be set here
      if self.working_dir is None:
          self.working_dir = Path(os.getcwd())
      if self.labeled_data_path is None:
          self.labeled_data_path = Path(self.working_dir) / 'labeled'
      if self.custom_classifier_path is None:
          self.custom_classifier_path = Path(self.working_dir) / 'custom_classifier_v02'
      # if these values have been supplied as strings, convert to Path objects
      self.working_dir = Path(self.working_dir)
      self.labeled_data_path = Path(self.labeled_data_path)
      self.custom_classifier_path = Path(self.custom_classifier_path)
      self.embeddings_path = Path(self.embeddings_path)
      if self.a2o_auth_token is not None:
          # allow a2o_auth_token instead of baw_auth_token,
          # for backwards compatibility
          self.baw_auth_token = self.a2o_auth_token
      




class AgileModeling:

  # user config for agile modeling
  config = None

  # bootstrap config, project state, and optional separator model
  bootstrap_config = None
  project_state = None
  separator = None

  # shortcut to bootstrap_config.model_config values
  sample_rate = None

  # search query
  query_audio = None
  sep_outputs = None

  # search results (for both single item and classifier searches)
  results = None
  all_scores = None

  # merged dataset. This is called 'merged' because it's a merged
  # dataset of new labelled examples and existing ones. 
  merged = None

  # custom classifier model
  wrapped_model: interface.LogitsOutputHead = None


  def __init__(self, config: AgileModelingConfig):
    self.config = config
    self.setup_bootstrap_config()
 
  
  def setup_bootstrap_config(self):
    """
    Depending on the source of the embeddings, loads a bootsrap config from
    the config associated with the embeddings. 
    @param config: AgileModelingConfig

    @return: bootstrap.BootstrapState
    """


    if self.config.data_source == 'a2o':
      embedding_config = baw_utils.get_a2o_embeddings_config()
      bootstrap_config = bootstrap.BootstrapConfig.load_from_embedding_config(
          embedding_config=embedding_config,
          annotated_path=self.config.labeled_data_path,
          embeddings_glob = '*/embeddings-*')
      self.config.embeddings_path = embedding_config.output_dir
    elif (self.config.embeddings_path
          or (Path(self.config.working_dir) / 'embeddings/config.json').exists()):
      if not self.config.embeddings_path:
        # Use the default embeddings path, as it seems we found a config there.
        self.config.embeddings_path = Path(self.config.working_dir) / 'embeddings'
      # Get relevant info from the embedding configuration.
      bootstrap_config = bootstrap.BootstrapConfig.load_from_embedding_path(
          embeddings_path=self.config.embeddings_path,
          annotated_path=self.config.labeled_data_path)
      # if the loaded embeddings config tells us it used separation model
      # set things up to use it for everything here
      if (bootstrap_config.model_key == 'separate_embed_model'
          and not self.config.separation_model_path.strip()):
        self.config.separation_model_key = 'separator_model_tf'
        self.config.separation_model_path = bootstrap_config.model_config.separator_model_tf_config.model_path
    else:
      raise ValueError('No embedding configuration found.')

    project_state = bootstrap.BootstrapState(
        bootstrap_config, baw_auth_token=self.config.baw_auth_token, baw_domain=self.config.baw_domain)

    # Load separation model.
    if self.config.separation_model_path:
      separation_config = config_dict.ConfigDict({
          'model_path': self.config.separation_model_path,
          'frame_size': 32000,
          'sample_rate': 32000,
      })
      separator = models.model_class_map()[
          self.config.separation_model_key].from_config(separation_config)
      print('Loaded separator model at {}'.format(self.config.separation_model_path))
    else:
      print('No separation model loaded.')
      separator = None

    self.bootstrap_config = bootstrap_config
    self.project_state = project_state
    self.separator = separator

  def load_classifier(self):
    """
    Load the classifier model from the path specified in the config.
    """
    cfg = config_dict.ConfigDict({
        'model_path': self.config.custom_classifier_path,
        'logits_key': 'custom',
    })
    self.wrapped_model = interface.LogitsOutputHead.from_config(cfg)
    

  def load_query_audio(self, 
                       audio_path,
                       start_s = 0):
    """
    Initializes the search query with the audio file path, bootstrap config,
    project state, and separator model.
    @param audio_path: str

    You may specify:
    * an audio filepath (like `/home/me/audio/example.wav`),
    * a Xeno-Canto id (like `xc12345`), or
    * an audio file URL (like
    https://upload.wikimedia.org/wikipedia/commons/7/7c/Turdus_merula_2.ogg).

    @param start_s: float
    Choose the start time for the audio window within the file.
    We will focus on the model's `window_size_s` seconds of audio,
    starting from `start_s` seconds into the file.

    """

    window_s = self.bootstrap_config.model_config['window_size_s']
    sample_rate = self.bootstrap_config.model_config['sample_rate']
    audio = audio_utils.load_audio(audio_path, sample_rate)

    # Display the full file.
    display.plot_audio_melspec(audio, sample_rate)

    # Display the selected window.
    print('-' * 80)
    print('Selected audio window:')
    st = int(start_s * sample_rate)
    end = int(st + window_s * sample_rate)
    if end > audio.shape[0]:
      end = audio.shape[0]
      st = max([0, int(end - window_s * sample_rate)])
    audio_window = audio[st:end]
    display.plot_audio_melspec(audio_window, sample_rate)

    query_audio = audio_window
    sep_outputs = None

    if self.separator is not None:
      sep_outputs = self.separator.embed(audio_window)

      for c in range(sep_outputs.separated_audio.shape[0]):
        print(f'Channel {c}')
        display.plot_audio_melspec(sep_outputs.separated_audio[c, :], sample_rate)
        print('-' * 80)
    else:
      sep_outputs = None
      print('No separation model loaded.')

    self.query_audio = query_audio
    self.sep_outputs = sep_outputs
    # this last one is probably not necessary as it's available in bootstrap_config
    # but this matches the original implementation in the notebook
    self.sample_rate = sample_rate


  def get_query_embedding(self, 
                          query_channel = -1):
    """
    @param query_label: str; Choose a name for the class.
    @param query_channel: int; If you have applied separation, choose a channel.
    @param audio_window: np.ndarray; The audio window to embed.
    """

    if query_channel < 0 or self.sep_outputs is None:
      query_audio = self.query_audio
    else:
      query_audio = self.sep_outputs.separated_audio[query_channel].copy()

    display.plot_audio_melspec(query_audio, self.sample_rate)

    outputs = self.project_state.embedding_model.embed(query_audio)
    query = outputs.pooled_embeddings('first', 'first')

    return query
  

  def do_search(self,
                query,
                top_k=50,
                metric='mip',
                target_score=None,
                random_sample=False,
                quit_after=500):
    
    """
    Run Top-K Search.
    @param top_k: int; Number of search results to capture.
    @param metric: str; Target distance for search results.
    @param target_score: float; This lets us try to hone in on a 'classifier boundary' instead of
    just looking at the closest matches. Set to 'None' for raw 'best results' search
    @param random_sample: bool; Set to 'None' for raw 'best results' search.
    """


    ds = self.project_state.create_embeddings_dataset(shuffle_files=True)


    results, all_scores = search.search_embeddings_parallel(
        ds, query,
        hop_size_s=self.bootstrap_config.embedding_hop_size_s,
        top_k=top_k, target_score=target_score, score_fn=metric,
        random_sample=random_sample)
    
    self.results = results
    self.all_scores = all_scores
  

  def plot_histogram_of_distances(self, 
                                  target_score=None, 
                                  metric='mip'):
    """
    Plot a histogram of the distances.
    """

    ys, _, _ = plt.hist(self.all_scores, bins=128, density=True)
    hit_scores = [r.score for r in self.results.search_results]
    plt.scatter(hit_scores, np.zeros_like(hit_scores), marker='|',
                color='r', alpha=0.5)

    plt.xlabel(metric)
    plt.ylabel('density')
    if target_score is not None:
      plt.plot([target_score, target_score], [0.0, np.max(ys)], 'r:')
      # Compute the proportion of scores < target_score
      hit_percentage = (self.all_scores < target_score).mean()
      print(f'score < target_score percentage : {hit_percentage:5.3f}')
    min_score = np.min(self.all_scores)
    plt.plot([min_score, min_score], [0.0, np.max(ys)], 'g:')

    plt.show()


  def save_validated_results(self):
    self.results.write_labeled_data(self.bootstrap_config.annotated_path,
                           self.project_state.embedding_model.sample_rate)


  def embed_labelled_set(self, 
                         time_pooling='mean'):

    merged = data_lib.MergedDataset.get_merged_dataset(
        base_dir=self.config.labeled_data_path,
        embedding_model=self.project_state.embedding_model,
        time_pooling=time_pooling,
        load_audio=False,
        target_sample_rate=-2,
        audio_file_pattern="*",
        embedding_config_hash=self.bootstrap_config.embedding_config_hash(),
        label_csv = self.config.label_csv
    )

    # Label distribution
    lbl_counts = np.sum(merged.data['label_hot'], axis=0)
    print('num classes :', (lbl_counts > 0).sum())
    print('mean ex / class :', lbl_counts.sum() / (lbl_counts > 0).sum())
    print('min ex / class :', (lbl_counts + (lbl_counts == 0) * 1e6).min())

    self.merged = merged


  def train_custom_classifier(self, 
                              train_ratio=0.9,
                              train_examples_per_class=None,
                              num_seeds=1,
                              batch_size=32,
                              num_epochs=128,
                              num_hiddens=-1,
                              learning_rate=1e-3):
    
    """
    Train small model over embeddings.
    @param train_ratio: float; Number of random training examples to choose form each class.
    @param train_examples_per_class: int; Number of random training examples to choose form each class.
                                          Set exactly one of `train_ratio` and `train_examples_per_class`.
    @param num_seeds: int; Number of random re-trainings. Allows judging model stability.
    @param batch_size: int; Classifier training hyperparams.
    @param num_epochs: int; Classifier training hyperparams.
    @param num_hiddens: int; Classifier training hyperparams.
    @param learning_rate: float; Classifier training hyperparams.

    """

    metrics = collections.defaultdict(list)
    for seed in tqdm.tqdm(range(num_seeds)):
      if num_hiddens > 0:
        model = classify.get_two_layer_model(
            num_hiddens, self.merged.embedding_dim, self.merged.num_classes, batch_norm=True, dtype=self.bootstrap_config.tensor_dtype)
      else:
        model = classify.get_linear_model(
            self.merged.embedding_dim, self.merged.num_classes, self.bootstrap_config.tensor_dtype)
      run_metrics = classify.train_embedding_model(
          model, self.merged, train_ratio, train_examples_per_class,
          num_epochs, seed, batch_size, learning_rate)
      metrics['acc'].append(run_metrics.top1_accuracy)
      metrics['auc_roc'].append(run_metrics.auc_roc)
      metrics['cmap'].append(run_metrics.cmap_value)
      metrics['maps'].append(run_metrics.class_maps)
      metrics['test_logits'].append(run_metrics.test_logits)
      if('all_test_logits' in run_metrics.__dict__):
        # all test logits added to produce precision recall plots
        # but may not be present in all versions of the code
        metrics['all_test_logits'].append(run_metrics.all_test_logits)

    mean_acc = np.mean(metrics['acc'])
    mean_auc = np.mean(metrics['auc_roc'])
    mean_cmap = np.mean(metrics['cmap'])
    # Merge the test_logits into a single array.
    test_logits = {
        k: np.concatenate([logits[k] for logits in metrics['test_logits']])
        for k in metrics['test_logits'][0].keys()
    }

    if 'all_test_logits' in metrics:
      # all test logits added to produce precision recall plots
      # but may not be present in all versions of the code
      # Merge the all_test_logits into a single array for each label.
      all_test_logits = {
          k: np.concatenate([logits[k] for logits in metrics['all_test_logits']])
          for k in metrics['all_test_logits'][0].keys()
      }
    else:
      all_test_logits = None

    print(f'acc:{mean_acc:5.2f}, auc_roc:{mean_auc:5.2f}, cmap:{mean_cmap:5.2f}')
    for lbl, auc in zip(self.merged.labels, run_metrics.class_maps):
      if np.isnan(auc):
        continue
      print(f'\n{lbl:8s}, auc_roc:{auc:5.2f}')
      colab_utils.prstats(f'test_logits({lbl})',
                          test_logits[self.merged.labels.index(lbl)])
      
    wrapped_model = interface.LogitsOutputHead(
      model_path=self.config.custom_classifier_path.as_posix(),
      logits_key='logits',
      logits_model=model,
      class_list=namespace.ClassList('custom', self.merged.labels),
    )
      
    self.wrapped_model = wrapped_model

    metrics = {
        'mean_acc': mean_acc,
        'mean_auc': mean_auc,
        'mean_cmap': mean_cmap,
        'test_logits': test_logits,
        'all_test_logits': all_test_logits,
    }
      
    return metrics
  

  def do_search_with_model(self,
                           num_results = 10,
                           target_class = None,
                           target_logit = 0,):
    """
    Run model on target unlabeled data.
    @param target_class: str; Choose the target class to work with.
    @param target_logit: int; Choose a target logit; will display results close to the target.
                            Set to None to get the highest-logit examples.
    @param num_results: int; Number of results to display.
    """

    if target_class is None:
      # use the first item in the merged labels
      # except for 'unknown' or 'neg' or 'negative'
      target_class = next((lbl for lbl in self.merged.labels if lbl not in ['unknown', 'neg', 'negative']), None)

    if target_class is None:
        raise ValueError(f"Please specify a valid target class, one of {self.merged.labels}")
     
    
    embeddings_ds = self.project_state.create_embeddings_dataset(
        shuffle_files=True)
    target_class_idx = self.merged.labels.index(target_class)
    results, all_logits = search.classifer_search_embeddings_parallel(
        embeddings_classifier=self.wrapped_model,
        target_index=target_class_idx,
        embeddings_dataset=embeddings_ds,
        hop_size_s=self.bootstrap_config.embedding_hop_size_s,
        target_score=target_logit,
        top_k=num_results
    )

    # Plot the histogram of logits.
    ys, _, _ = plt.hist(all_logits, bins=128, density=True)
    plt.xlabel(f'{target_class} logit')
    plt.ylabel('density')
    # plt.yscale('log')
    plt.plot([target_logit, target_logit], [0.0, np.max(ys)], 'r:')
    plt.show()

    self.results = results
    self.all_scores = all_logits
  

  def display_search_results(self,
                             display_labels = None,
                             extra_labels = ['unknown', 'neg'], 
                             samples_per_page=25):
    """
    Display the search results for both single item and classifier searches.
    """
    if display_labels is None:
      display_labels = self.merged.labels
    elif  isinstance(display_labels, str):
      display_labels = [display_labels]

    #@markdown Specify any extra labels you would like displayed.
    for label in extra_labels:
      if label not in display_labels:
        display_labels += (label,)

    page_state = display.PageState(
        np.ceil(len(self.results.search_results) / samples_per_page))

    display.display_paged_results(
        self.results, page_state, samples_per_page,
        project_state=self.project_state,
        embedding_sample_rate=self.project_state.embedding_model.sample_rate,
        exclusive_labels=False,
        checkbox_labels=display_labels,
        max_workers=5,
    )
                                    

  def save_model(self):

    self.wrapped_model.save_model(
        self.config.custom_classifier_path,
        self.config.embeddings_path)
    

  def run_inference(self,
                    output_filepath = None,
                    default_threshold = 0.0, 
                    class_thresholds = None, 
                    include_classes = [],
                    exclude_classes = ['unknown', 'neg', 'negative']):
    """
    runs inference over all embeddings, and writes results that meet the threshold to a CSV file.
    @param output_filepath: str; Path to write the CSV file.
    @param default_threshold: float; Default threshold for all classes.
    @param class_thresholds: dict; Dictionary of class thresholds.
    @param include_classes: list; List of classes to include. Ignored if empty
    @param exclude_classes: list; List of classes to exclude.

    If default threshold is None, all logits are written. This can lead to very large CSV files.
    If 
    """
    #@title Write classifier inference CSV. { vertical-output: true }

    #@markdown This cell writes detections (locations of audio windows where
    #@markdown the logit was greater than a threshold) to a CSV file.

    if self.wrapped_model is None:
      self.load_classifier()
    
    if output_filepath is None:
      output_filepath = Path(self.config.working_dir) / 'inference.csv'
    
    print(f'Writing to {output_filepath}')


    if default_threshold is None:
      # In this case, all logits are written. This can lead to very large CSV files.
      class_thresholds = None
    elif class_thresholds is None:
      # In this case, use the default threshold for all classes
      class_thresholds = collections.defaultdict(lambda: default_threshold)
    else:
      # in this case use the default threshold for all classes not in the class_thresholds dict
      class_thresholds = collections.defaultdict(lambda: default_threshold, class_thresholds)


    embeddings_ds = self.project_state.create_embeddings_dataset(
        shuffle_files=True)
    classify.write_inference_csv(
        embeddings_ds=embeddings_ds,
        model=self.wrapped_model,
        # labels=self.merged.labels was changed to this, in the case that we just load the classifier without training,
        labels=self.wrapped_model.class_list.classes,
        output_filepath=output_filepath,
        threshold=class_thresholds,
        embedding_hop_size_s=self.bootstrap_config.embedding_hop_size_s,
        include_classes=include_classes,
        exclude_classes=exclude_classes)
    
  
  def prepare_call_density_estimation(self, 
                                      target_class: str = None,
                                      bounds: List[float] = [0.0, 0.9, 0.99, 0.999, 1.0],
                                      samples_per_bin = 25):
    """
    Prepares things for the call density estimation
    @param target_class: str; Choose the target class to work with.
    @param bounds: List[float]; List of quantiles to use for binning.
    @param samples_per_bin: int; Number of samples to use per bin.

    Searches the embeddings for a given number of examples per logit bin, 
    with the bins defined by the quantiles in `bounds`.


    """

    bounds = np.array(bounds)
    num_bins = len(bounds) - 1

    # Select `top_k`` so that we are reasonably sure to get at least samples_per_bin
    # samples in the rarest bin in a randomly selected set of `top_k` examples.
    bin_probs = bounds[1:] - bounds[:-1]
    rarest_prob = np.min(bin_probs)
    top_k = int(samples_per_bin  / rarest_prob * 2)

    embeddings_ds = self.project_state.create_embeddings_dataset(shuffle_files=True)
    results, all_logits = search.classifer_search_embeddings_parallel(
        embeddings_classifier=self.wrapped_model.logits_head,
        target_index=self.wrapped_model.class_list.classes.index(target_class),
        random_sample=True,
        top_k=top_k,
        hop_size_s=self.bootstrap_config.embedding_hop_size_s,
        embeddings_dataset=embeddings_ds,
    )

    q_bounds = np.quantile(all_logits, bounds)
    binned = [[] for _ in range(num_bins)]
    for r in results.search_results:
      result_bin = np.argmax(r.score < q_bounds) - 1
      binned[result_bin].append(r)
    binned = [np.random.choice(b, samples_per_bin, replace=False) for b in binned]

    combined_results = []
    for b in binned:
      combined_results.extend(b)
    rng = np.random.default_rng(42)
    rng.shuffle(combined_results)

    ys, _, _, = plt.hist(all_logits, bins=100, density=True)
    for q in q_bounds:
      plt.plot([q, q], [0.0, np.max(ys)], 'k:', alpha=0.75)
    plt.show()

    binned_validation_examples = {
      'target_class': target_class,
      'combined_results': combined_results,
      'q_bounds': q_bounds,
      'bin_probs': bin_probs,
      'num_bins': num_bins
    }

    return binned_validation_examples

  def write_validation_log(self, binned_validation_examples):

    target_class = binned_validation_examples['target_class']
    combined_results = binned_validation_examples['combined_results']
    q_bounds = binned_validation_examples['q_bounds']
    bin_probs = binned_validation_examples['bin_probs']
  
    validation_log_filepath = (
        Path(self.config.working_dir) / f'validation_{target_class}.csv')

    filenames = []
    timestamp_offsets = []
    scores = []
    is_pos = []
    weights = []
    bins = []

    for r in combined_results:
      if not r.label_widgets: continue
      value = r.label_widgets[0].value
      if value is None:
        continue
      filenames.append(r.filename)
      scores.append(r.score)
      timestamp_offsets.append(r.timestamp_offset)

      # Get the bin number and sampling weight for the search result.
      result_bin = np.argmax(r.score < q_bounds) - 1
      bins.append(result_bin)
      weights.append(bin_probs[result_bin])

      if value == target_class:
        is_pos.append(1)
      elif value == f'not {target_class}':
        is_pos.append(-1)
      elif value == 'unsure':
        is_pos.append(0)

    label = [target_class for _ in range(len(filenames))]

    validation_log = {
        'filenames': filenames,
        'timestamp_offsets': timestamp_offsets,
        'scores': scores,
        'is_pos': is_pos,
        'weights': weights,
        'bins': bins,
    }

    log = pd.DataFrame(validation_log)
    log.to_csv(validation_log_filepath, mode='a')

    return validation_log


def plot_precision_recall_curves(test_logits_dict, test_labels):
    """Plots Precision-Recall curves for each label using logits, true labels, and true label array."""

    from sklearn.metrics import precision_recall_curve, average_precision_score
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))  
    true_classes = test_logits_dict.pop('true_label')  # Extract and remove 'true_label'

    # Micro-averaged curve
    # all_logits = np.concatenate(list(test_logits_dict.values()), axis=0)
    # all_true_labels = np.repeat(list(test_logits_dict.keys()), [len(v) for v in test_logits_dict.values()])

    # precision_micro, recall_micro, _ = precision_recall_curve(
    #     all_true_labels == true_classes[:, np.newaxis], all_logits
    # )
    # ap_micro = average_precision_score(
    #     all_true_labels == true_classes[:, np.newaxis], all_logits, average="micro"
    # )
    # plt.plot(recall_micro, precision_micro, linestyle='--', label=f'Micro-averaged (AP = {ap_micro:.2f})')

    for label_index in test_logits_dict.keys():
        label_name = test_labels[label_index]
        if label_name == 'neg' or label_name == 'unknown' or label_name == 'negative':
            continue
        logits = test_logits_dict[label_index]
        
        # Use true labels directly
        y_true = (true_classes == label_index).astype(int) 
        
        precision, recall, _ = precision_recall_curve(y_true, logits)
        ap = average_precision_score(y_true, logits)
        plt.plot(recall, precision, label=f'Label: {label_name} (AP = {ap:.2f})')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves for Each Label")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.4) 
    plt.show()
    
def estimate_call_density(binned_validation_examples, validation_log):
  """
  Estimates call density based on validated binned results
  """

  target_class = binned_validation_examples['target_class']
  combined_results = binned_validation_examples['combined_results']
  q_bounds = binned_validation_examples['q_bounds']
  bin_probs = binned_validation_examples['bin_probs']
  num_bins = binned_validation_examples['num_bins']
  scores = validation_log['scores']
  is_pos = validation_log['is_pos']

  # Collect validated labels by bin.
  bin_pos = [0 for i in range(num_bins)]
  bin_neg = [0 for i in range(num_bins)]
  for score, pos in zip(scores, is_pos):
    result_bin = np.argmax(r.score < q_bounds) - 1
    if pos == 1:
      bin_pos[result_bin] += 1
    elif pos == -1:
      bin_neg[result_bin] += 1

  # Create beta distributions.
  prior = 0.1
  betas = [scipy.stats.beta(p + prior, n + prior)
          for p, n in zip(bin_pos, bin_neg)]
  # MLE positive rate in each bin.
  mle_b = np.array([bin_pos[b] / (bin_pos[b] + bin_neg[b] + 1e-6)
                    for b in range(num_bins)])

  # Probability of each bin, P(b).
  p_b = bin_probs

  # MLE total call density.
  q_mle = np.dot(mle_b, p_b)

  num_beta_samples = 10_000
  q_betas = []
  for _ in range(num_beta_samples):
    qs_pos = np.array([b.rvs(size=1)[0] for b in betas])  # P(+|b)
    q_beta = np.dot(qs_pos, p_b)
    q_betas.append(q_beta)

  # Plot call density estimate.
  plt.figure(figsize=(10, 5))
  xs, ys, _ = plt.hist(q_betas, density=True, bins=25, alpha=0.25)
  plt.plot([q_mle, q_mle], [0.0, np.max(xs)], 'k:', alpha=0.75,
          label='q_mle')

  low, high = np.quantile(q_betas, [0.05, 0.95])
  plt.plot([low, low], [0.0, np.max(xs)], 'g', alpha=0.75, label='low conf')
  plt.plot([high, high], [0.0, np.max(xs)], 'g', alpha=0.75, label='high conf')

  plt.xlim(0.0, 1.0)
  plt.xlabel('Call Rate (q)')
  plt.ylabel('P(q)')
  plt.title(f'Call Density Estimation ({target_class})')
  plt.legend()
  plt.show()

  print(f'MLE Call Density: {q_mle:.4f}')
  print(f'(Low/MLE/High) Call Density Estimate: ({low:5.4f} / {q_mle:5.4f} / {high:5.4f})')


def estimate_roc_auc(binned_validation_results, validation_log):
  """
  Naive Estimation of ROC-AUC for target class
  @param binned_validation_results: dict; result of prepare_call_density_estimateion

  Computes ROC-AUC from the validation logs, with bin weighting
  ROC-AUC is the overall probability that a random positive example
  has a higher classifier score than a random negative example.
  """

  num_bins = binned_validation_results['num_bins']
  scores = validation_log['scores']
  is_pos = validation_log['is_pos']

  # Probability of bins
  p_b = [1.0 / 2**(k + 1) for k in range(num_bins - 1)]
  p_b.append(p_b[-1])
  p_b = np.array(p_b)

  bin_pos, bin_neg = np.array(bin_pos), np.array(bin_neg)
  # Compute P(z(+) > z(-))
  # P(b_i|+) * P(b_k|-)
  n_pos = np.sum(bin_pos)
  n_neg = np.sum(bin_neg)
  p_pos_b = np.array(bin_pos) / (bin_pos + bin_neg)
  p_neg_b = np.array(bin_neg) / (bin_pos + bin_neg)
  p_pos = np.sum(p_pos_b * p_b)
  p_neg = np.sum(p_neg_b * p_b)

  p_b_pos = p_pos_b * p_b / p_pos
  p_b_neg = p_neg_b * p_b / p_neg
  roc_auc = 0
  # For off-diagonal bin pairs:
  # Take the probability of drawing a pos from bin j and neg from bin i.
  # If j > i, all pos examples are scored higher, so contributes directly to the
  # total ROC-AUC.
  for i in range(num_bins):
    for j in range(i + 1, num_bins):
      roc_auc += p_b_pos[j] * p_b_neg[i]

  # For diagonal bin-pairs:
  # Look at actual in-bin observations for diagonal contribution.
  bins = np.array(bins)
  is_pos = np.array(is_pos)

  for b in range(num_bins):
    bin_pos_idxes = np.argwhere((bins == b) * (is_pos == 1))[:, 0]
    bin_neg_idxes = np.argwhere((bins == b) * (is_pos == -1))[:, 0]
    bin_pos_scores = np.array(scores)[bin_pos_idxes]
    bin_neg_scores = np.array(scores)[bin_neg_idxes]
    if bin_pos_scores.size == 0:
      continue
    if bin_neg_scores.size == 0:
      continue
    # Count total number of pairs where the pos examples have a higher score than
    # a negative example.
    hits = ((bin_pos_scores[:, np.newaxis]
            - bin_neg_scores[np.newaxis, :]) > 0).sum()
    bin_roc_auc = hits / (bin_pos_scores.size * bin_neg_scores.size)
    # Contribution is the probability of pulling both pos and neg examples
    # from this bin, multiplied by the bin's ROC-AUC.
    roc_auc += bin_roc_auc * p_b_pos[b] * p_b_neg[b]

  print(f'ROC-AUC : {roc_auc:.3f}')