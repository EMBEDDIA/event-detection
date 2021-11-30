# -*- coding: utf-8 -*-
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers import BertConfig
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
)
#from transformers.modeling_outputs import (
#    BaseModelOutputWithCrossAttentions,
#    BaseModelOutputWithPoolingAndCrossAttentions,
#    CausalLMOutputWithCrossAttentions,
#    MaskedLMOutput,
#    MultipleChoiceModelOutput,
#    NextSentencePredictorOutput,
#    QuestionAnsweringModelOutput,
#    SequenceClassifierOutput,
#    TokenClassifierOutput,
#)
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers import BertPreTrainedModel, BertModel
from transformers.utils import logging

from modules.transformer import TransformerEncoder
from transformers import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from transformers import AutoConfig
from transformers import PretrainedConfig

after_norm=True
n_heads=6
head_dims=128
num_layers=2
attn_type='transformer'
trans_dropout=0.45
d_model = n_heads * head_dims
dim_feedforward = int(2 * d_model)
dropout=0.45
device = torch.device("cuda:0")

class BertForQuestionAnswering(BertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)#, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.transformer = TransformerEncoder(num_layers, d_model, n_heads, dim_feedforward, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=attn_type == 'transformer', dropout_attn=None,
                                              pos_embed='sin')
        self.init_weights()


    #@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    #@add_code_sample_docstrings(
    #    tokenizer_class=_TOKENIZER_FOR_DOC,
    #    checkpoint="bert-base-uncased",
    #    output_type=QuestionAnsweringModelOutput,
    #    config_class=_CONFIG_FOR_DOC,
    #)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        argument_label_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        
#        import pdb;pdb.set_trace()
        sequence_output = self.transformer(sequence_output, attention_mask) 

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



