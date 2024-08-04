# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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

import os
import warnings
from pathlib import Path
from typing import Optional, Union

# from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding

from .feature_extraction_carir import CarirFeatureExtractor, RtfFeatureExtractor

CLASS_TO_MODULE_MAP = {
    "feature_extractor": CarirFeatureExtractor,
    "CarirFeatureExtractor": CarirFeatureExtractor,
    "rtf_extractor": RtfFeatureExtractor,
    "RtfFeatureExtractor": RtfFeatureExtractor,
}


class CarirProcessor():

    attributes = ["feature_extractor", "rtf_extractor"]
    # Names need to be attr_class for attr in attributes
    feature_extractor_class = None
    rtf_extractor_class = None

    feature_extractor_class = "CarirFeatureExtractor"
    rtf_extractor_class = "RtfFeatureExtractor"

    def __init__(self, *args, **kwargs):
        # Sanitize args and kwargs
        for key in kwargs:
            if key not in self.attributes:
                raise TypeError(f"Unexpected keyword argument {key}.")
        for arg, attribute_name in zip(args, self.attributes):
            if attribute_name in kwargs:
                raise TypeError(f"Got multiple values for argument {attribute_name}.")
            else:
                kwargs[attribute_name] = arg

        if len(kwargs) != len(self.attributes):
            raise ValueError(
                f"This processor requires {len(self.attributes)} arguments: {', '.join(self.attributes)}. Got "
                f"{len(args)} arguments instead."
            )

        # Check each arg is of the proper class (this will also catch a user initializing in the wrong order)
        for attribute_name, arg in kwargs.items():
            class_name = getattr(self, f"{attribute_name}_class")
            setattr(self, attribute_name, arg)

    def __call__(self, rtfs=None, audios=None, return_tensors=None, **kwargs):
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to RobertaTokenizerFast's [`~RobertaTokenizerFast.__call__`] if `text` is not `None` to
        encode the text. To prepare the audio(s), this method forwards the `audios` and `kwrags` arguments to
        doctsring of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audios (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The audio or batch of audios to be prepared. Each audio can be NumPy array or PyTorch tensor. In case
                of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels,
                and T the sample length of the audio.

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **audio_features** -- Audio features to be fed to a model. Returned when `audios` is not `None`.
        """
        sampling_rate = kwargs.pop("sampling_rate", None)

        if rtfs is None and audios is None:
            raise ValueError("You have to specify either text or audios. Both cannot be none.")

        if rtfs is not None:
            encoding = self.rtf_extractor(rtfs, return_tensors=return_tensors, **kwargs)

        if audios is not None:
            audio_features = self.feature_extractor(
                    audios, sampling_rate=sampling_rate, return_tensors=return_tensors, **kwargs)

        if rtfs is not None and audios is not None:
            encoding["input_audio_features"] = audio_features.input_features
            return encoding
        elif rtfs is not None:
            return encoding
        else:
            return BatchEncoding(data=dict(**audio_features), tensor_type=return_tensors)


    @classmethod
    def from_json_path(cls,
                       json_path,
                       **kwargs,
                       ):
        import json
        config = json.load(open(json_path, 'r'))
        modules = []
        for class_type in attributes:
            modules.append(CLASS_TO_MODULE_MAP[class_type](**config))
        return cls(*modules)



    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to RobertaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        assert 0
        # return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to RobertaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        assert 0
        # return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        rtf_input_names = self.rtf_extractor.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(rtf_input_names + feature_extractor_input_names))

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        os.makedirs(save_directory, exist_ok=True)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        # if self._auto_class is not None:
        #     attrs = [getattr(self, attribute_name) for attribute_name in self.attributes]
        #     configs = [(a.init_kwargs if isinstance(a, PreTrainedTokenizerBase) else a) for a in attrs]
        #     custom_object_save(self, save_directory, config=configs)

        for attribute_name in self.attributes:
            attribute = getattr(self, attribute_name)
            # Include the processor class in the attribute config so this processor can then be reloaded with the
            # `AutoProcessor` API.
            if hasattr(attribute, "_set_processor_class"):
                attribute._set_processor_class(self.__class__.__name__)
            attribute.save_pretrained(save_directory)

        # if self._auto_class is not None:
        #     # We added an attribute to the init_kwargs of the tokenizers, which needs to be cleaned up.
        #     for attribute_name in self.attributes:
        #         attribute = getattr(self, attribute_name)
        #         if isinstance(attribute, PreTrainedTokenizerBase):
        #             del attribute.init_kwargs["auto_map"]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ):
        r"""
        Instantiate a processor associated with a pretrained model.

        <Tip>

        This class method is simply calling the feature extractor
        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`], image processor
        [`~image_processing_utils.ImageProcessingMixin`] and the tokenizer
        [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`] methods. Please refer to the docstrings of the
        methods above for more information.

        </Tip>

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~SequenceFeatureExtractor.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            **kwargs
                Additional keyword arguments passed along to both
                [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`] and
                [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`].
        """
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(*args)

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoProcessor"):
        """
        Register this class with a given auto class. This should only be used for custom feature extractors as the ones
        in the library are already mapped with `AutoProcessor`.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoProcessor"`):
                The auto class to register this new feature extractor with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        args = []
        for attribute_name in cls.attributes:
            class_name = getattr(cls, f"{attribute_name}_class")
            if isinstance(class_name, tuple):
                classes = tuple(getattr(transformers_module, n) if n is not None else None for n in class_name)
                use_fast = kwargs.get("use_fast", True)
                if use_fast and classes[1] is not None:
                    attribute_class = classes[1]
                else:
                    attribute_class = classes[0]
            else:
                attribute_class = CLASS_TO_MODULE_MAP[class_name]

            args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs))
        return args
