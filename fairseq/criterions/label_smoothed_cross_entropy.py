# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import logging


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        simpleforvisual=False
    ):
        if not simpleforvisual:
            super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True, training_type=None, co_T=None, ma_T=None, loss_weight=None, new_loss_cal=None, margin_delta_co=1, margin_delta_su=1):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # logging.info('sample = {}'.format(sample['id']))
        assert reduce, 'reduce is not always True! '
        # print('sample[id]', sample['sample2']['id'])
        # print('sample[id]', sample['sample']['id'])
        if training_type == 'co':
            sample2 = sample['sample2']
            sample = sample['sample']
            # print('sample[id]', sample['id'])
            # print('sample[id]', sample2['id'])
            score1 = model(**sample["net_input"], output_score_only=True, classification_head_name='coherence_score')
        elif training_type == 'ma':
            if model.args.ma_rmcomp:
                sample = sample['sample']
                net_output1 = model(**sample["net_input"])
            else:
                sample2 = sample['sample2']
                sample = sample['sample']
                # print('sample[id]', sample['id'])
                # print('sample[id]', sample2['id'])
                net_output1 = model(**sample["net_input"])
                net_output2 = model(**sample2["net_input"])
        else:  # 'or', 'fi', 'su', 'fo'
            # print('sample[id]', sample['id'])
            net_output = model(**sample["net_input"])
        assert not self.sentence_avg
        if training_type == 'co':
            if not new_loss_cal:  # coherence, old loss
                score2 = model(**sample2['net_input'], output_score_only=True, classification_head_name='coherence_score')
                loss = torch.nn.functional.softmax(torch.cat([score1, score2], dim=1) / co_T, dim=1)
                s1_soft = loss[:, 0].mean().item()
                s2_soft = loss[:, 1].mean().item()
                loss = margin_delta_co - (loss[:, 0] - loss[:, 1])
                loss[loss < 0] = 0
                loss = loss.mean()  # 'co_ps', 'co_ns', 'co_ps_soft', 'co_ns_soft', 'co_loss',
                # 'ma_l1', 'ma_l2', 'ma_l1_soft', 'ma_l2_soft', 'ma_loss'
                other_log = {
                    'aux_task': 'co',
                    'co_ps': score1.mean().item(),
                    'co_ns': score2.mean().item(),
                    'co_ps_soft': s1_soft,
                    'co_ns_soft': s2_soft,
                    'co_loss': loss.item()
                }
                loss = loss * loss_weight
                sample_size = (
                    sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
                )
            else:  # coherence, new loss
                co_sample_size = sample['id'].shape[0]
                score2 = model(**sample2['net_input'], output_score_only=True, classification_head_name='coherence_score')
                loss = torch.nn.functional.softmax(torch.cat([score1, score2], dim=1) / co_T, dim=1)
                s1_soft = loss[:, 0].sum().item()
                s2_soft = loss[:, 1].sum().item()
                loss = margin_delta_co - (loss[:, 0] - loss[:, 1])
                loss[loss < 0] = 0
                loss = loss.sum()  # loss = loss.mean()
                other_log = {
                    'aux_task': 'co',
                    'co_ps': (score1.sum().item(), co_sample_size),
                    'co_ns': (score2.sum().item(), co_sample_size),
                    'co_ps_soft': (s1_soft, co_sample_size),
                    'co_ns_soft': (s2_soft, co_sample_size),
                    'co_loss': (loss.item(), co_sample_size)
                }
                loss = loss * loss_weight  # placed below to get normalized log about the ma_loss
                sample_size = co_sample_size
                # print('=' * 10, 'coherence', co_sample_size)
                # print('coherence new loss, %s/%s, mean: %s' % (other_log['co_loss'][0], other_log['co_loss'][1], other_log['co_loss'][0] / other_log['co_loss'][1]))
        elif training_type == 'ma':
            if model.args.ma_rmcomp:  # sub-summary, old & new loss
                loss1, _ = self.compute_loss(model, net_output1, sample, reduce=reduce)
                loss = loss1
                sample_size = (
                    sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
                )
                if not new_loss_cal:
                    other_log = {
                        'aux_task': 'ma',
                        'ma_loss': loss.item()
                    }
                else:
                    other_log = {
                        'aux_task': 'ma',
                        'ma_loss': (loss.item(), sample_size),
                    }
                loss = loss * loss_weight
                    # print('sub-summary new loss, %s/%s, mean: %s' % (other_log['ma_loss'][0], other_log['ma_loss'][1], other_log['ma_loss'][0] / other_log['ma_loss'][1]))
            else:  # margin sub-summary
                if not new_loss_cal:  # margin sub-summary, old loss
                    loss1, _ = self.compute_loss(model, net_output1, sample, reduce=reduce)
                    loss2, _ = self.compute_loss(model, net_output2, sample2, reduce=reduce)
                    loss_cat = torch.cat([loss1.unsqueeze(0), loss2.unsqueeze(0)], dim=0)
                    score_ma = torch.nn.functional.softmax(loss_cat / ma_T, dim=0)
                    loss = margin_delta_su - (score_ma[1] - score_ma[0])
                    other_log = {
                        'aux_task': 'ma',
                        'ma_l1': loss1.mean().item(),
                        'ma_l2': loss2.mean().item(),
                        'ma_l1_soft': score_ma[0].mean().item(),
                        'ma_l2_soft': score_ma[1].mean().item(),
                        'ma_loss': loss.item()
                    }
                    loss = loss * loss_weight
                    sample_size = (
                        sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
                    )
                    # print('ma, not model.args.ma_rmcomp, %s/%s, mean: %s' % (other_log['ma_loss'][0], other_log['ma_loss'][1], other_log['ma_loss'][0] / other_log['ma_loss'][1]))
                else:  # margin sub-summary, new loss
                    ma_sample_size = sample["ntokens"]
                    ma_sentence_size = sample['id'].shape[0]
                    loss1, _ = self.compute_loss(model, net_output1, sample, reduce=True)
                    loss2, _ = self.compute_loss(model, net_output2, sample2, reduce=True)
                    loss_cat = torch.cat([(loss1 / ma_sample_size).unsqueeze(0), (loss2 / ma_sample_size).unsqueeze(0)], dim=0)
                    score_ma = torch.nn.functional.softmax(loss_cat / ma_T, dim=0)
                    loss = margin_delta_su - (score_ma[1] - score_ma[0])
                    loss = loss * ma_sentence_size
                    other_log = {
                        'aux_task': 'ma',
                        'ma_l1': ((loss1 / ma_sample_size).item(), ma_sample_size),
                        'ma_l2': ((loss2 / ma_sample_size).item(), ma_sample_size),
                        'ma_l1_soft': (score_ma[0].item(), 1),
                        'ma_l2_soft': (score_ma[1].item(), 1),
                        'ma_loss': (loss.item(), ma_sentence_size)
                    }
                    loss = loss * loss_weight
                    sample_size = ma_sentence_size
                    # print('='*10, 'margin', ma_sentence_size)
                    # print('ma, not model.args.ma_rmcomp, %s/%s, mean: %s' % (other_log['ma_loss'][0], other_log['ma_loss'][1], other_log['ma_loss'][0] / other_log['ma_loss'][1]))
        else:  # 'or', 'fi', 'su', 'fo'
            assert training_type not in ['or', 'fi', 'su']
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
            # print('fo, %s/%s, mean: %s' % (loss.item(), sample_size, loss.item() / sample_size))
        # input('wait')
        if training_type in ['co', 'or', 'fi', 'su', 'ma']:
            return loss, sample_size, None, other_log
        else:  # 'fo'
            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
            if self.report_accuracy:
                n_correct, total = self.compute_accuracy(model, net_output, sample)
                logging_output["n_correct"] = utils.item(n_correct.data)
                logging_output["total"] = utils.item(total.data)
            return loss, sample_size, logging_output, None

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
