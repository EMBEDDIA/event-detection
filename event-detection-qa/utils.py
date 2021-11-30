# -*- coding: utf-8 -*-

import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from transformers.data.processors.squad import (squad_convert_example_to_features_init,
                                                squad_convert_example_to_features)
from transformers.file_utils import is_tf_available, is_torch_available
import torch
if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf
from tqdm import tqdm
def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    padding_strategy="max_length",
    return_dataset=False,
    threads=1,
    tqdm_enabled=True,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model. It is
    model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.
    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        padding_strategy: Default to "max_length". Which padding strategy to use
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset, if 'tf': returns a tf.data.Dataset
        threads: multiple processing threads.
    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`
    Example::
        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)
        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """
    # Defining helper methods
    features = []

    threads = min(threads, cpu_count())
    with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
                disable=not tqdm_enabled,
            )
        )

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

        if not is_training:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask
            )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_is_impossible,
            )
#        import pdb;pdb.set_trace()

        return features, dataset
    elif return_dataset == "tf":
        if not is_tf_available():
            raise RuntimeError("TensorFlow must be installed to return a TensorFlow dataset.")

        def gen():
            for i, ex in enumerate(features):
                if ex.token_type_ids is None:
                    yield (
                        {
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "feature_index": i,
                            "qas_id": ex.qas_id,
                        },
                        {
                            "start_positions": ex.start_position,
                            "end_positions": ex.end_position,
                            "cls_index": ex.cls_index,
                            "p_mask": ex.p_mask,
                            "is_impossible": ex.is_impossible,
                        },
                    )
                else:
                    yield (
                        {
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "token_type_ids": ex.token_type_ids,
                            "feature_index": i,
                            "qas_id": ex.qas_id,
                        },
                        {
                            "start_positions": ex.start_position,
                            "end_positions": ex.end_position,
                            "cls_index": ex.cls_index,
                            "p_mask": ex.p_mask,
                            "is_impossible": ex.is_impossible,
                        },
                    )

        # Why have we split the batch into a tuple? PyTorch just has a list of tensors.
        if "token_type_ids" in tokenizer.model_input_names:
            train_types = (
                {
                    "input_ids": tf.int32,
                    "attention_mask": tf.int32,
                    "token_type_ids": tf.int32,
                    "feature_index": tf.int64,
                    "qas_id": tf.string,
                },
                {
                    "start_positions": tf.int64,
                    "end_positions": tf.int64,
                    "cls_index": tf.int64,
                    "p_mask": tf.int32,
                    "is_impossible": tf.int32,
                },
            )

            train_shapes = (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                    "feature_index": tf.TensorShape([]),
                    "qas_id": tf.TensorShape([]),
                },
                {
                    "start_positions": tf.TensorShape([]),
                    "end_positions": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                    "is_impossible": tf.TensorShape([]),
                },
            )
        else:
            train_types = (
                {"input_ids": tf.int32, "attention_mask": tf.int32, "feature_index": tf.int64, "qas_id": tf.string},
                {
                    "start_positions": tf.int64,
                    "end_positions": tf.int64,
                    "cls_index": tf.int64,
                    "p_mask": tf.int32,
                    "is_impossible": tf.int32,
                },
            )

            train_shapes = (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "feature_index": tf.TensorShape([]),
                    "qas_id": tf.TensorShape([]),
                },
                {
                    "start_positions": tf.TensorShape([]),
                    "end_positions": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                    "is_impossible": tf.TensorShape([]),
                },
            )

        return tf.data.Dataset.from_generator(gen, train_types, train_shapes)
    else:
        return features