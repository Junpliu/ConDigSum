# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import encoders
from fairseq.hub_utils import GeneratorHubInterface
from omegaconf import open_dict


logger = logging.getLogger(__name__)


class BARTHubInterface(GeneratorHubInterface):
    """A simple PyTorch Hub interface to BART.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/bart
    """

    def __init__(self, cfg, task, model):
        super().__init__(cfg, task, [model])
        self.model = self.models[0]

    def encode(
        self, sentence: str, *addl_sentences, no_separator=True, no_bpe=False
    ) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`).

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> bart.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> bart.encode(' world').tolist()
            [0, 232, 2]
            >>> bart.encode('world').tolist()
            [0, 8331, 2]
        """
        if not no_bpe:
            tokens = self.bpe.encode(sentence)
        else:
            tokens = sentence
        if len(tokens.split(" ")) > min(self.max_positions) - 2:
            tokens = " ".join(tokens.split(" ")[: min(self.max_positions) - 2])
        bpe_sentence = "<s> " + tokens + " </s>"
        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator else ""
            bpe_sentence += " " + self.bpe.encode(s) + " </s>"
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
        return tokens.long()

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = tokens == self.task.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [
            self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences
        ]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def _build_sample(self, src_tokens: List[torch.LongTensor]):
        # assert torch.is_tensor(src_tokens)
        dataset = self.task.build_dataset_for_inference(
            src_tokens,
            [x.numel() for x in src_tokens],
        )
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(lambda tensor: tensor.to(self.device), sample)
        return sample

    def generate(self, tokens: List[torch.LongTensor], beam: int = 5, verbose: bool = False, **kwargs) -> torch.LongTensor:
        # rewrite
        # inference_step_args = inference_step_args or {}
        # bsz = len(tokens)
        # inference_step_args["prefix_tokens"] = tokens[0].new_full(
        #         (bsz, 1), fill_value=self.task.source_dictionary.bos()).to(device=self.device)
        # # build generator using current args as well as any kwargs
        # gen_args = copy.deepcopy(self.cfg.generation)
        # with open_dict(gen_args):
        #     gen_args.beam = beam
        #     for k, v in kwargs.items():
        #         setattr(gen_args, k, v)
        # generator = self.task.build_generator(self.models, gen_args)
        #
        # inference_step_args = inference_step_args or {}
        # results = []
        # for batch in self._build_batches(tokenized_sentences, skip_invalid_size_inputs):
        #     batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
        #     translations = self.task.inference_step(
        #         generator, self.models, batch, **inference_step_args
        #     )
        #     for id, hypos in zip(batch["id"].tolist(), translations):
        #         results.append((id, hypos))
        #
        # # sort output to match input order
        # outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]
        # return outputs
        ########################################3

        sample = self._build_sample(tokens)
        # build generator using current args as well as any kwargs
        # print('cfg = ', self.cfg)
        gen_args = copy.copy(self.cfg.generation)
        gen_args.beam = beam
        for k, v in kwargs.items():
            if k not in ['topN']:
                setattr(gen_args, k, v)
        generator = self.task.build_generator([self.model], gen_args)
        translations = self.task.inference_step(
            generator,
            [self.model],
            sample,
            prefix_tokens=sample['net_input']['src_tokens'].new_zeros((len(tokens), 1)).fill_(
                self.task.source_dictionary.bos())
        )

        if verbose:
            src_str_with_unk = self.string(tokens)
            logger.info('S\t{}'.format(src_str_with_unk))

        def getarg(name, default):
            return getattr(gen_args, name, getattr(self.args, name, default))

        # Process top predictions
        # print('top predictions = ', sum([len(x) for x in translations]) / 819)
        # print('need top %s predictions, and we have %s now. ' % (kwargs['topN'], len(translations[0])))
        # for item in translations[0]:
        #     print('first sample = ', item)
        # input('wait')
        if kwargs['topN'] == 1:
            hypos = [x[0] for x in translations]
        else:
            hypos = [x[:kwargs['topN']] for x in translations]
        hypos = [v for _, v in sorted(zip(sample['id'].tolist(), hypos))]
        return hypos

    def sample(
        self, sentences: List[str], beam: int = 1, verbose: bool = False, **kwargs
    ) -> List[str]:
        kwargs['topN'] = 1
        inputs = [self.encode(sentence, no_bpe=True) for sentence in sentences]

        hypos = self.generate(inputs, beam, verbose, **kwargs)
        if kwargs['topN'] == 1:
            return [self.decode(x['tokens']) for x in hypos]
        else:
            ret_lst = []
            for x_lst in hypos:
                ret_lst.append(' || '.join([self.decode(x['tokens']) for x in x_lst]))
            return ret_lst

    def get_utter_feature(self, sentence: str, **kwargs):
        assert kwargs.get('dataset', None).startswith('SAMSumInd') or kwargs.get('dataset', None).startswith('mediasum')
        sentences = [sentence]
        token_count = len(sentence.split())
        utter_count = len(sentence.split(' 39811 '))
        token_option = np.where(np.array(sentence.split()) == '39811')[0].tolist() + [token_count]
        utter_idx = []
        last = 0
        while len(utter_idx) != utter_count:
            end = token_option[len(utter_idx)]
            utter_idx.append((last, end))
            last = end

        inputs = [self.encode(sentence, no_bpe=True) for sentence in sentences]
        kwargs.pop('dataset')

        sample = self._build_sample(inputs)
        with torch.no_grad():
            encoder_output = self.model(**sample["net_input"], output_encoder_hidden_only=True)
        encoder_output = encoder_output[0][1:-1]
        utter_vector = []
        for start, end in utter_idx:
            utter_v = (encoder_output[start:end].sum(axis=0) / (end - start)).numpy()
            utter_vector.append(utter_v)
        return utter_vector


    def visualize_coherence(
        self, sentence: str, **kwargs
    ):
        sentences = [sentence]
        if kwargs.get('dataset', None).startswith('SAMSumInd'):
            inputs = [self.encode(sentence, no_bpe=True) for sentence in sentences]
            kwargs.pop('dataset')
        elif kwargs.get('dataset', None) == 'SAMSumSep':
            inputs = [self.encode(sentence) for sentence in sentences]
            kwargs.pop('dataset')
        elif kwargs.get('dataset', None).startswith('mediasum'):
            inputs = [self.encode(sentence, no_bpe=True) for sentence in sentences]
            kwargs.pop('dataset')
        else:
            assert kwargs.get('dataset', None) is None or kwargs.get('dataset', None).startswith('SAMSum')
            inputs = [self.encode(sentence) for sentence in sentences]
        sample = self._build_sample(inputs)
        score = self.model(**sample["net_input"], output_score_only=True, classification_head_name='coherence_score')
        assert len(score) == 1
        return score[0].item()

    def visualize_subsummary(
            self, sentence: str, target: str, **kwargs
    ):
        if getattr(self, "criterion", None) is None:
            import importlib
            criter = importlib.import_module("fairseq.criterions.label_smoothed_cross_entropy")
            self.criterion = criter.LabelSmoothedCrossEntropyCriterion(None, False, 0.1, simpleforvisual=True)
            self.criterion.padding_idx = 1
        sentences = [sentence]
        assert len(sentences) == 1, 'only one sentence is allowed here avoiding the order of sentences being meesed up! '
        if kwargs.get('dataset', None).startswith('SAMSumInd'):
            inputs = [self.encode(sentence, no_bpe=True) for sentence in sentences]
            kwargs.pop('dataset')
        elif kwargs.get('dataset', None) == 'SAMSumSep':
            inputs = [self.encode(sentence) for sentence in sentences]
            kwargs.pop('dataset')
        elif kwargs.get('dataset', None).startswith('mediasum'):
            inputs = [self.encode(sentence, no_bpe=True) for sentence in sentences]
            kwargs.pop('dataset')
        else:
            assert kwargs.get('dataset', None) is None or kwargs.get('dataset', None).startswith('SAMSum')
            inputs = [self.encode(sentence) for sentence in sentences]
        sample = self._build_sample(inputs)
        # print('sample = ', sample)
        sample['target'] = target.unsqueeze(dim=0)
        sample['net_input']['prev_output_tokens'] = torch.tensor([2] + sample['target'][0].tolist()[:-1]).unsqueeze(dim=0)
        # print('sample = ', sample)
        net_output = self.model(**sample["net_input"])
        loss, _ = self.criterion.compute_loss(self.model, net_output, sample, reduce=True)
        return loss.item()

    def extract_features(
        self, tokens: torch.LongTensor, return_all_hiddens: bool = False
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > min(self.model.max_positions()):
            raise ValueError(
                "tokens exceeds maximum length: {} > {}".format(
                    tokens.size(-1), self.model.max_positions()
                )
            )
        tokens.to(device=self.device),
        prev_output_tokens = tokens.clone()

        prev_output_tokens[:, 0] = tokens.gather(
            1,
            (tokens.ne(self.task.source_dictionary.pad()).sum(dim=1) - 1).unsqueeze(-1),
        ).squeeze()

        prev_output_tokens[:, 1:] = tokens[:, :-1]
        features, extra = self.model(
            src_tokens=tokens,
            src_lengths=None,
            prev_output_tokens=prev_output_tokens,
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra["inner_states"]
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def register_classification_head(
        self, name: str, num_classes: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_classification_head(
            name, num_classes=num_classes, embedding_size=embedding_size, **kwargs
        )

    def predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        features = self.extract_features(tokens.to(device=self.device))
        sentence_representation = features[
            tokens.eq(self.task.source_dictionary.eos()), :
        ].view(features.size(0), -1, features.size(-1))[:, -1, :]

        logits = self.model.classification_heads[head](sentence_representation)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)

    def fill_mask(
        self,
        masked_inputs: List[str],
        topk: int = 5,
        match_source_len: bool = True,
        **generate_kwargs
    ):
        masked_token = '<mask>'
        batch_tokens = []
        for masked_input in masked_inputs:
            assert masked_token in masked_input, \
                "please add one {} token for the input".format(masked_token)

            text_spans = masked_input.split(masked_token)
            text_spans_bpe = (' {0} '.format(masked_token)).join(
                [self.bpe.encode(text_span.rstrip()) for text_span in text_spans]
            ).strip()
            tokens = self.task.source_dictionary.encode_line(
                '<s> ' + text_spans_bpe + ' </s>',
                append_eos=False,
                add_if_not_exist=False,
            ).long()
            batch_tokens.append(tokens)

        # ensure beam size is at least as big as topk
        generate_kwargs['beam'] = max(
            topk,
            generate_kwargs.get('beam', -1),
        )
        generate_kwargs['match_source_len'] = match_source_len
        batch_hypos = self.generate(batch_tokens, **generate_kwargs)

        return [
            [(self.decode(hypo['tokens']), hypo['score']) for hypo in hypos[:topk]]
            for hypos in batch_hypos
        ]
