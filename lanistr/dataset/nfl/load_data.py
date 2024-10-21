"""Copyright 2024 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from typing import Any, Dict, List, Union

import numpy as np
import omegaconf
import pandas as pd
import torch
from torch.utils import data
import transformers
from utils.data_utils import MaskGenerator


def load_nfl(
    args: omegaconf.DictConfig, tokenizer: transformers.AutoTokenizer
) -> Dict[str, Union[data.Dataset, Dict[str, Any]]]:
    """Load the NFL dataset.

    Args:
        args: The arguments for the experiment.
        tokenizer: The tokenizer to use for the text.

    Returns:
        A dictionary containing the train, valid, and test datasets.
    """
    categorical_cols = ['team', 'player', 'position']
    numerical_cols = ['age', 'height', 'weight']
    nfl_data = load_multimodal_data(args)
    nfl_data, cat_idxs, cat_dims, input_dim = preprocess_nfl_tabular_features(
        data=nfl_data,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
    )
    feature_names = categorical_cols + numerical_cols
    text_names = ['news']
    time_series_names = ['odds', 'play_by_play']
    train_data, test_data, valid_data = get_train_and_test_splits(args, nfl_data)

    tabular_data_information = {
        'input_dim': input_dim,
        'cat_idxs': cat_idxs,
        'cat_dims': cat_dims,
        'feature_names': feature_names,
        'text_names': text_names,
        'time_series_names': time_series_names,
    }

    dataframes = {
        'train': train_data,
        'valid': valid_data,
        'test': test_data,
        'tabular_data_information': tabular_data_information,
    }
    dataset = create_multimodal_dataset_from_dataframes(args, dataframes, tokenizer)
    return dataset


def create_multimodal_dataset_from_dataframes(
    args: omegaconf.DictConfig,
    dataframes: Dict[str, pd.DataFrame],
    tokenizer: transformers.BertTokenizer,
) -> Dict[str, Union[data.Dataset, Dict[str, Any]]]:
    """Create a multimodal dataset from dataframes.

    Args:
        args: The arguments for the experiment.
        dataframes: The dataframes to use for the dataset.
        tokenizer: The tokenizer to use for the text.

    Returns:
        A dictionary containing the train, valid, and test datasets.
    """
    train_transform, test_transform = get_nfl_transforms(args)

    mm_train = NFLDataset(
        args=args,
        df=dataframes['train'],
        tokenizer=tokenizer,
        transform=train_transform,
        feature_names=dataframes['tabular_data_information']['feature_names'],
        text_names=dataframes['tabular_data_information']['text_names'],
        time_series_names=dataframes['tabular_data_information']['time_series_names'],
        text=args.text,
        time=args.time,
        tab=args.tab,
    )
    mm_test = NFLDataset(
        args=args,
        df=dataframes['test'],
        tokenizer=tokenizer,
        transform=test_transform,
        feature_names=dataframes['tabular_data_information']['feature_names'],
        text_names=dataframes['tabular_data_information']['text_names'],
        time_series_names=dataframes['tabular_data_information']['time_series_names'],
        text=args.text,
        time=args.time,
        tab=args.tab,
    )
    mm_val = NFLDataset(
        args=args,
        df=dataframes['valid'],
        tokenizer=tokenizer,
        transform=train_transform,
        feature_names=dataframes['tabular_data_information']['feature_names'],
        text_names=dataframes['tabular_data_information']['text_names'],
        time_series_names=dataframes['tabular_data_information']['time_series_names'],
        text=args.text,
        time=args.time,
        tab=args.tab,
    )

    return {
        'train': mm_train,
        'valid': mm_val,
        'test': mm_test,
        'tabular_data_information': dataframes['tabular_data_information'],
    }


class NFLDataset(data.Dataset):
    """NFL dataset with text, tabular data, and time series data."""

    def __init__(
        self,
        args: omegaconf.DictConfig,
        df: pd.DataFrame,
        tokenizer: transformers.BertTokenizer,
        transform: Any,
        feature_names: List[str],
        text_names: List[str],
        time_series_names: List[str],
        text: bool,
        time: bool,
        tab: bool,
    ):
        """Initialize the NFLDataset.

        Args:
            args: The arguments for the experiment.
            df: The dataframe to use for the dataset.
            tokenizer: The tokenizer to use for the text.
            transform: The transform to use for the images.
            feature_names: The names of the features columns.
            text_names: The names of the text columns.
            time_series_names: The names of the time series columns.
            text: Whether to use text.
            time: Whether to use time series data.
            tab: Whether to use tabular data.
        """
        self.args = args
        self.df = df
        self.df = self.df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        if tab:
            self.features = self.df[feature_names].values

        if text:
            self.texts = df[text_names].values

        if time:
            self.time_series = df[time_series_names].values

        self.mask_generator = MaskGenerator(
            input_size=args.image_size,
            mask_patch_size=args.mask_patch_size,
            model_patch_size=args.model_patch_size,
            mask_ratio=args.image_masking_ratio,
        )

        self.text = text
        self.time = time
        self.tab = tab

    def __getitem__(self, index: int):
        """Get the item at the given index.

        Args:
            index: The index of the item to get.

        Returns:
            The item at the given index.
        """
        row = self.df.iloc[index]

        item = {}

        # text
        if self.text:
            input_ids_list = []
            attention_mask_list = []
            for text in self.texts[index]:
                encode_result = self.encode_text(text)
                input_ids_list.append(encode_result['input_ids'])
                attention_mask_list.append(encode_result['attention_mask'])
            # input_ids has shape (text_num, token_length)
            item['input_ids'] = torch.cat(input_ids_list)
            # attention_mask has shape (text_num, token_length)
            item['attention_mask'] = torch.cat(attention_mask_list)

        # tabular
        if self.tab:
            item['features'] = torch.tensor(
                np.vstack(self.features[index]).astype(np.float32)
            ).squeeze(1)

        # time series
        if self.time:
            item['time_series'] = torch.tensor(
                np.vstack(self.time_series[index]).astype(np.float32)
            ).squeeze(1)

        # ground truth label if finetuning
        if self.args.task == 'finetune':
            item['labels'] = torch.tensor(row['labels'], dtype=torch.long)

        return item

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.df)

    def encode_text(self, text: str):
        try:
            return self.tokenizer.encode_plus(
                text,
                max_length=self.args.max_token_length,
                truncation=True,
                add_special_tokens=True,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(e)
            return self.tokenizer.encode_plus(
                '',
                max_length=self.args.max_token_length,
                truncation=True,
                add_special_tokens=True,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
            )
