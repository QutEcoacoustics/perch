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

"""Handlers for the Bio-Acoustics Workbench (BAW/A2O) API."""

import io
import os
import re
from typing import Generator, Sequence
import urllib

from chirp import audio_utils
import librosa
from ml_collections import config_dict
import numpy as np
import requests
import soundfile

# regex to extract a recording id from a canonical file name
FILE_ID_TO_UID_PATTERN = re.compile(r".*_(\d+).[^\.]+$")

# regex to extract the domain and recording id from a full recording URL
FULL_RECORDING_PATTERN = re.compile(r"https://(.+?)/audio_recordings/(\d+)/original")

BAW_DOMINS_ALLOWLIST = ["api.acousticobservatory.org", "api.ecosounds.org", "api.staging.ecosounds.org"]

# Bit of a hack to provide website listen links for BAW audio segments
WEBSITE_DOMAIN_MAP = {
    "api.acousticobservatory.org": "data.acousticobservatory.org",
    "api.ecosounds.org": "www.ecosounds.org",
}

def make_baw_audio_url_from_file_id(
    file_id: str,
    offset_s: float,
    window_size_s: float,
    baw_domain: str = "api.acousticobservatory.org",
    extension: str = "flac"
) -> str:
  """Construct an baw audio URL."""
  # Extract the recording UID. Example:
  # 'site_0277/20210428T100000+1000_Five-Rivers-Dry-A_909057.flac' -> 909057
  # 'site_0277/20210428T100000+1000_Five-Rivers-Dry-A_909057.wav' -> 909057

  domain, arid = extract_arid_and_domain(file_id)
  if domain is None:
    domain = baw_domain

  return make_baw_audio_url_from_arid(
      arid=arid,
      offset_s=offset_s,
      window_size_s=window_size_s,
      baw_domain=domain,
      extension=extension
  )

def make_baw_audio_url_from_arid(
    arid: str,
    offset_s: float,
    window_size_s: float,
    baw_domain: str = "api.acousticobservatory.org",
    extension: str = "flac",
    check_baw_domains = False
) -> str:
  """Construct an baw media query URL."""
  if check_baw_domains and baw_domain not in BAW_DOMINS_ALLOWLIST:
    raise ValueError(f"Invalid domain: {baw_domain}. Valid domains are: {BAW_DOMINS_ALLOWLIST}")

  offset_s = int(offset_s)
  # See: https://api.staging.ecosounds.org/api-docs/index.html
  audio_path = f"https://{baw_domain}/audio_recordings/{arid}/media.{extension}"
  if offset_s <= 0 and window_size_s <= 0:
    return audio_path
  params = {}
  if offset_s > 0:
    params["start_offset"] = offset_s
  if window_size_s > 0:
    params["end_offset"] = offset_s + int(window_size_s)
  audio_path = audio_path + "?" + urllib.parse.urlencode(params)
  return audio_path


def make_baw_listen_link_from_arid(
    arid: str,
    offset_s: float,
    window_size_s: float,
    baw_domain: str = "data.acousticobservatory.org",
    check_baw_domains = False
) -> str:
  """Construct an baw media query URL."""
  if check_baw_domains and baw_domain not in BAW_DOMINS_ALLOWLIST:
    raise ValueError(f"Invalid domain: {baw_domain}. Valid domains are: {BAW_DOMINS_ALLOWLIST}")

  offset_s = int(offset_s)
  # See: https://api.staging.ecosounds.org/api-docs/index.html
  if baw_domain in WEBSITE_DOMAIN_MAP:
    baw_domain = WEBSITE_DOMAIN_MAP[baw_domain]
  audio_path = f"https://{baw_domain}/listen/{arid}/"
  if offset_s <= 0 and window_size_s <= 0:
    return audio_path
  params = {}
  if offset_s > 0:
    params["start"] = offset_s
  if window_size_s > 0:
    params["end"] = offset_s + int(window_size_s)
  audio_path = audio_path + "?" + urllib.parse.urlencode(params)
  return audio_path


def extract_arid_and_domain(file_id: str) -> tuple[str, str]:
  """Extract the domain and recording ID from a full recording url or a canonical file name."""
  match = FULL_RECORDING_PATTERN.match(file_id)
  if match:
    domain = match.group(1)
    arid = int(match.group(2))  # Convert arid to an integer
    return domain, arid
  
  match = FILE_ID_TO_UID_PATTERN.match(file_id)
  if match:
    domain = None
    arid = int(match.group(1))  # Convert arid to an integer
    return domain, arid

  if not match:
    raise ValueError(f"Invalid file_id format: {file_id}")
  


def load_baw_audio(
    audio_url: str,
    auth_token: str,
    sample_rate: int,
    session: requests.Session,
) -> np.ndarray | None:
  """Load audio from the Bioacoustics Workbench API.

  Args:
    audio_url: URL to load the audio from.
    auth_token: The BAW API auth token.
    sample_rate: The sample rate to resample the audio to.
    session: The requests session to use.

  Returns:
    The audio as a numpy array, or None if the audio could not be loaded.
  """

  if session is None:
    # Use requests.get instead of session.get if no session is provided.
    session = requests
  headers={}
  if auth_token:
    headers["Authorization"] = f"Token token={auth_token}"
      
  audio_response = session.get(
      url=audio_url,
      headers=headers,
  )
  if not audio_response.ok:
    print(audio_response.status_code)
    return None

  # Load the audio and resample.
  try:
    with io.BytesIO(audio_response.content) as f:
      sf = soundfile.SoundFile(f)
      audio = sf.read()
      audio = librosa.resample(
          audio, orig_sr=sf.samplerate, target_sr=sample_rate
      )
  except soundfile.LibsndfileError:
    return None
  return audio


def multi_load_baw_audio(
    filepaths: Sequence[str],
    offsets: Sequence[int],
    auth_token: str = None,
    sample_rate: int = 32000,
    **kwargs,
) -> Generator[np.ndarray, None, None]:
  """Creates a generator that loads audio from the BAW API."""
  session = requests.Session()
  session.mount(
      "https://",
      requests.adapters.HTTPAdapter(
          max_retries=requests.adapters.Retry(total=5, backoff_factor=0.5)
      ),
  )
  baw_audio_loader = lambda fp, offset: load_baw_audio(
      fp, sample_rate=sample_rate, auth_token=auth_token, session=session
  )
  iterator = audio_utils.multi_load_audio_window(
      filepaths=filepaths,
      offsets=offsets,
      audio_loader=baw_audio_loader,
      **kwargs,
  )
  try:
    for ex in iterator:
      yield ex
  finally:
    session.close()


def get_a2o_embeddings_config() -> config_dict.ConfigDict:
  """Returns an embeddings config for the public A2O embeddings."""
  chirp_public_bucket = "gs://chirp-public-bucket"
  perch_512_model_path = os.path.join(chirp_public_bucket, "models/perch_4_512")
  embeddings_uri = os.path.join(
      chirp_public_bucket, "embeddings/a2o_embeddings_perch512"
  )

  config = config_dict.ConfigDict({
      "output_dir": embeddings_uri,
      "source_file_patterns": [
          "https://api.acousticobservatory.org/audio_recordings/download/flac/*",
      ],
      "num_shards_per_file": 1,
      "shard_len_s": 60,
      "start_shard_idx": 0,
      "num_direct_workers": 8,
      "embed_fn_config": {
          "write_embeddings": True,
          "write_logits": False,
          "write_separated_audio": False,
          "write_raw_audio": False,
          "file_id_depth": 1,
          "model_key": "taxonomy_model_tf",
          "tensor_dtype": "float16",
          "model_config": {
              "model_path": perch_512_model_path,
              "window_size_s": 5.0,
              "hop_size_s": 5.0,
              "sample_rate": 32000,
          },
          "logits_head_config": {
              "model_path": perch_512_model_path + "/speech_empty_filter",
              "logits_key": "nuisances",
              "channel_pooling": "",
          },
      },
  })
  return config
