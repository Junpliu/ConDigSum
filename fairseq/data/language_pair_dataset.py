# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils
import random
from fairseq_cli.utils import *
# from fairseq_cli.utils import SEPidx, MASKidx, MASKSentidx


logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
    sort=True
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if (
            alignment[:, 0].max().item() >= src_len - 1
            or alignment[:, 1].max().item() >= tgt_len - 1
        ):
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True
        )
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1.0 / align_weights.float()

    id = torch.LongTensor([s["id"] for s in samples]).to(device=samples[0]['source'].device)
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    ).to(device=samples[0]['source'].device)
    if sort:
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
    # logger.info(src_tokens.device)
    # logger.info(sort_order.device)
        src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        # logger.info(sort_order)
        # logger.info(target)
        if sort:
            target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).to(device=samples[0]['source'].device)
        if sort:
            tgt_lengths = tgt_lengths.index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        ) if sort else prev_output_tokens

    if samples[0].get("alignment", None) is not None:
        logger.info('if samples[0].get("alignment", None) is not None:')
        input('error')
        bsz, tgt_sz = batch["target"].shape
        src_sz = batch["net_input"]["src_tokens"].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += torch.arange(len(sort_order), dtype=torch.long) * tgt_sz
        if left_pad_source:
            offsets[:, 0] += src_sz - src_lengths
        if left_pad_target:
            offsets[:, 1] += tgt_sz - tgt_lengths

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(
                sort_order, offsets, src_lengths, tgt_lengths
            )
            for alignment in [samples[align_idx]["alignment"].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch["alignments"] = alignments
            batch["align_weights"] = align_weights

    if samples[0].get("constraints", None) is not None:
        # Collate the packed constraints across the samples, padding to
        # the length of the longest sample.
        logger.info('if samples[0].get("constraints", None) is not None:')
        input('error')
        lens = [sample.get("constraints").size(0) for sample in samples]
        max_len = max(lens)
        constraints = torch.zeros((len(samples), max(lens))).long()
        for i, sample in enumerate(samples):
            constraints[i, 0 : lens[i]] = samples[i].get("constraints")
        batch["constraints"] = constraints

    return batch


class LanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1,
        seed=None,
        pre_task=None,  # 'None' for finetune, 'SR_TI' for pretraining under "sentence reordering" and "text infilling"
        mask_p=None,
        lam=None,
        aux_task=None,
        aux_p=None,
        aux_method=None,
        origindata=None
    ):
        logger.info('seed = {} {}'.format(seed, '=' * 30))
        logger.info('pre_task = {} {}'.format(pre_task, '=' * 30))
        logger.info('mask_p = {} {}'.format(mask_p, '=' * 30))
        logger.info('lam = {} {}'.format(lam, '=' * 30))
        logger.info('aux_task = {} {}'.format(aux_task, '=' * 30))
        logger.info('aux_p = {} {}'.format(aux_p, '=' * 30))
        logger.info('aux_method = {} {}'.format(aux_method, '=' * 30))
        logger.info('origindata = {} {}'.format(origindata, '=' * 30))
        assert pre_task is None or pre_task == 'SR_TI' and mask_p is not None and lam is not None or pre_task == 'MLM_GSG'
        assert aux_task is None or (aux_task == 'SR_TI' and aux_p is not None and mask_p is not None and lam is not None) or aux_task.islower() or aux_task == 'MLM_GSG'
        if aux_task is not None and aux_task.islower():
            for aux in aux_task.split('_'):
                assert aux in ['co', 'or', 'su', 'fi', 'ma']
        self.aux_task = aux_task
        self.aux_task_lst = aux_task.split('_') if aux_task is not None else []
        self.mask_p = mask_p
        self.lam = lam
        self.aux_p = aux_p
        self.pega_mask_sent = 0.3
        self.pega_mask_token = 0.15

        self.aux_method = aux_method
        if origindata is not None:
            assert False, 'never pass non-empty origindata here! '
            assert aux_method in ['sample', 'batch'] or pre_task == 'MLM_GSG'
            self.origindata = origindata
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(
                tgt
            ), "Source and target must contain the same number of examples"
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert (
                self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        self.pretrain_task = pre_task
        if pre_task is not None:
            assert seed is not None
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.error_count = 0
        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info("bucketing source lengths: {}".format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info(
                    "bucketing target lengths: {}".format(list(self.tgt.buckets))
                )

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        if self.pretrain_task is None:  # finetune / standard test phase / aux_task coherence
            if self.aux_task == 'SR_TI':
                p = np.random.random()
                if p < self.aux_p:
                    # print('mask')
                    src_item, tgt_item = self.get_maskspan_shuffle_pair(index)
                else:
                    tgt_item = self.tgt[index] if self.tgt is not None else None
                    src_item = self.src[index]
            elif self.aux_task is None or self.aux_task.islower():  # standard summarization test phase or simple finetune
                tgt_item = self.tgt[index] if self.tgt is not None else None
                src_item = self.src[index]
            # elif self.aux_task.islower() and self.aux_method == 'sample':  # 'co' sample
            #     p = np.random.random()
            #     if len(self.aux_task_lst) == 2:
            #         if p < self.aux_p / 2:
            #             ret = self.origindata.__getitem__(index, cur_sample_type=self.aux_task_lst[0], aux_method='sample')
            #             assert len(ret[0]) == 1
            #             src_item, tgt_item = ret[0][0]['source'], ret[1][0].get('target', None)
            #         elif p < self.aux_p:
            #             ret = self.origindata.__getitem__(index, cur_sample_type=self.aux_task_lst[1], aux_method='sample')
            #             assert len(ret[0]) == 1
            #             src_item, tgt_item = ret[0][0]['source'], ret[1][0].get('target', None)
            #         else:
            #             tgt_item = self.tgt[index] if self.tgt is not None else None
            #             src_item = self.src[index]
            #     elif len(self.aux_task_lst) == 1:
            #         if p < self.aux_p:
            #             logger.error('get aux sample')
            #             ret = self.origindata.__getitem__(index, cur_sample_type=self.aux_task_lst[0], aux_method='sample')
            #             assert len(ret[0]) == 1
            #             src_item, tgt_item = ret[0][0]['source'], ret[1][0].get('target', None)
            #             print('ret', tgt_item, src_item)
            #             assert False
            #         else:
            #             logger.error('get normal sample')
            #             tgt_item = self.tgt[index] if self.tgt is not None else None
            #             src_item = self.src[index]
            #             print('ret', tgt_item, src_item)
            #             assert False
            #     else:
            #         assert False, 'error! '
            #     assert False
        else:  # pretrain, SR_TI
            if self.pretrain_task == 'SR_TI':
                src_item, tgt_item = self.get_maskspan_shuffle_pair(index)
            # elif self.pretrain_task == 'MLM_GSG':
            #     src_item, tgt_item = self.get_mlm_gsg_pair(index)

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        assert not self.append_eos_to_target and not self.append_bos and not self.remove_eos_from_source
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]
        # logger.info('{} {} {} {}'.format(self.src_dict.bos(), self.src_dict.eos(), self.tgt_dict.bos(), self.tgt_dict.eos()))
        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
        }

        # logger.info('id {}'.format(index))
        # logger.info('src_item {}'.format(src_item))
        # logger.info('tgt_item {}'.format(tgt_item))
        #
        # input('load wait')
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        return example

    def shuffle_sents(self, sents, sep_id):
        # 1. remove the last "2" id
        # 2. seperate sentences based on "sep_id"
        # 3. shuffle sentences
        # (removed) 4. add "2" id to the end.
        sents = sents.tolist()
        if sents[-1] == 2:
            sents = sents[:-1]
        new_sents = []
        cur_sent = []
        sep_id_count = 0
        for item in sents:
            if item != sep_id:
                cur_sent.append(item)
            else:
                sep_id_count += 1
                new_sents.append(cur_sent)
                cur_sent = []
        new_sents.append(cur_sent)
        if len(new_sents) == 2:  # bug421, former cur_sent
            new_sents[0], new_sents[1] = new_sents[1], new_sents[0]
        else:
            random.shuffle(new_sents)
        assert sep_id_count == len(new_sents) - 1
        ret_sents = []
        for sen_idx, sen in enumerate(new_sents):
            ret_sents.extend(sen)
            if sen_idx != len(new_sents) - 1:
                ret_sents.append(sep_id)
        # ret_sents.append(2)
        # return torch.LongTensor(ret_sents)
        return ret_sents

    def get_maskspan_shuffle_pair(self, index):
        src_item = self.src[index]
        src_list = self.shuffle_sents(src_item, SEPidx)
        src_item, tgt_item = None, src_item
        # print('after sentence shuffle:', src_list)
        src_list = self.get_mask_span_pair(src_list)
        src_list.append(2)  # end of sequence !
        # print('after mask_span src_list = ', src_list)
        src_item = torch.LongTensor(src_list)
        return src_item, tgt_item

    def get_mask_span_pair(self, src_list):
        # print()
        # org_src = [self.src_dict[word] for word in src_list]
        # print('mask_span_pair src: ', src_list)

        length = len(src_list)  # don't mask the last 'eos'
        # source = list(range(length))

        mask_span_lst = []
        start = 0
        while start < length:
            p = np.random.random()
            if p < self.mask_p / self.lam:
                cur_len = np.random.poisson(lam=self.lam)
                end = min(start + cur_len, length)
                mask_span_lst.append((start, end))
                start = end
            else:
                start += 1
        cur_idx = 0
        cur_span_idx = 0
        new_source = []
        while cur_idx < length:
            if cur_span_idx < len(mask_span_lst) and cur_idx == mask_span_lst[cur_span_idx][0]:
                span_s, span_e = mask_span_lst[cur_span_idx]
                new_source.append(MASKidx)
                cur_idx = span_e
                cur_span_idx += 1
            else:
                if cur_span_idx < len(mask_span_lst):
                    next_idx = mask_span_lst[cur_span_idx][0]
                else:
                    next_idx = length
                for copy_idx in range(cur_idx, next_idx):
                    new_source.append(src_list[copy_idx])
                cur_idx = next_idx
        # print('new_source', new_source)
        # print('len(new_souce)', len(new_source))
        # print(length - word_mask_count + len(mask_span_lst))
        # ==========================

        # new_src = [self.src_dict[word] for word in source]
        # print('new src: ', new_src)
        # new_tgt = [self.src_dict[word] for word in target]
        # print('new tgt: ', new_tgt)
        return new_source

    # def get_mlm_gsg_pair(self, index):
    #     src_item = self.src[index]
    #     src_list = [str(item) for item in src_item.tolist()][:-1]
    #     src_tokens = ' '.join(src_list).split(' %s ' % str(SEPidx))
    #     # print('utterances_lst = ', src_tokens)
    #     sents = []
    #     for i, tokens in enumerate(src_tokens):
    #         sents.append([int(tok) for tok in tokens.split(' ')])
    #
    #     top_utter_idx_lst = self.origindata.__getitem__(index, cur_sample_type='MLM_GSG', aux_method=None)
    #     # print(len(top_utter_idx_lst), 'top_utter_idx_lst = ', top_utter_idx_lst)
    #     if len(sents) != len(top_utter_idx_lst):
    #         self.error_count += 1
    #         if self.error_count % 500 == 0:
    #             print(self.error_count)
    #         # print('dialog_idx', index)
    #         # print(len(sents))
    #         # print(len(top_utter_idx_lst))
    #     top_utter_idx_lst = [item[0] for item in top_utter_idx_lst[:max(int(len(sents) * self.pega_mask_sent), 1)]]
    #
    #     tgt_list = []
    #     for utter_idx in range(len(sents)):
    #         if utter_idx in top_utter_idx_lst:
    #             tgt_list.extend(sents[utter_idx])
    #             sents[utter_idx] = [MASKSentidx]
    #     # print('after sentence mask, tgt_lst', tgt_list)
    #     # print('after sentence mask, source_lst', sents)
    #     src_list = []
    #     for sent_idx, sent in enumerate(sents):
    #         src_list.extend(sent)
    #         if sent_idx != len(sents) - 1:
    #             src_list.append(SEPidx)
    #     # print('after sep, source_lst', src_list)
    #
    #     dont_mask = np.random.choice(list(range(len(src_list))), len(src_list) - int(len(src_list) * self.pega_mask_token), replace=False)
    #     dont_mask = sorted(dont_mask)
    #     # print('dont_mask', dont_mask)
    #     new_src_list = []
    #     for idx in range(len(src_list)):
    #         if idx in dont_mask:
    #             new_src_list.append(src_list[idx])
    #         else:
    #             new_src_list.append(MASKidx)
    #     # print('after sentence mask, source_lst', new_src_list)
    #
    #     new_src_list.append(2)
    #     src_item = torch.LongTensor(new_src_list)
    #     tgt_list.append(2)
    #     tgt_item = torch.LongTensor(tgt_list)
    #     return src_item, tgt_item

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        # logger.info('pad_to_length = {}'.format(pad_to_length))
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.src_sizes[indices]
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )
