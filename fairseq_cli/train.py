#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable

import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed_utils import is_master
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf

from fairseq_cli.utils import OriginDataset, AverageMeter


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | Line: %(lineno)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout
)
logger = logging.getLogger("fairseq_cli.train")


def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    if is_master(cfg.distributed_training) and "job_logging_cfg" in cfg:
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)
    print('cfg.model.co_window_size', cfg.model.co_window_size)
    print('='*100)
    print(task.src_dict)
    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in cfg.dataset.valid_subset.split(","):  # valid
        task.load_dataset(valid_sub_split, 1, False, seed=cfg.model.seed,
                          pre_task=cfg.model.pre_task,
                          mask_p=cfg.model.mask_p if cfg.model.pre_task is not None else None,
                          lam=cfg.model.lam if cfg.model.pre_task is not None else None)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    print('enter task.build_model()')
    print('cfg.model', cfg.model)
    print('cfg.model.aux_task', cfg.model.aux_task)
    assert cfg.model.loss_reduce in ['alter', 'sum', None], 'aux_task param error! '
    if cfg.model.loss_reduce is not None:
        assert cfg.model.aux_task is not None
    if cfg.model.loss_reduce is not None:
        assert cfg.model.aux_method == 'batch'
    if cfg.model.aux_task is not None:
        assert cfg.model.aux_task != ''
        for training_type in cfg.model.aux_task.split('_'):
            # 'fo' means formal summarization task
            # 'co': coherence, 'or': order, 'fi': fill-in-blank, 'su': sub_summary
            assert training_type in ['co', 'or', 'fi', 'su', 'ma', 'SR', 'TI']
    if cfg.model.aux_task is not None and 'SR_TI' in cfg.model.aux_task:
        assert cfg.model.aux_p is not None
    if cfg.model.pre_task is not None:
        assert cfg.task.data in ['SAMSumSep', 'mediasum', 'SAMSumInd', 'mediasumsm', 'SAMSumIndBase'], 'cfg.model.task.data = {}'.format(cfg.task.data)
        cfg.checkpoint.best_checkpoint_metric = 'nll_loss'
        assert cfg.checkpoint.best_checkpoint_metric == 'nll_loss'
    task_param = {
        'co': ['co_window_size', 'co_truncate', 'class_lr', 'co_T', 'co_sample_truncate'],
        'su': ['su_window_size', 'su_truncate'],
        'ma': ['ma_truncate', 'ma_version', 'ma_T', 'ma_sample_truncate']
    }
    if cfg.model.aux_task is not None:
        if 'su' in cfg.model.aux_task or 'ma' in cfg.model.aux_task:
            assert cfg.model.rouge_version is not None and cfg.model.rouge_metric is not None
        if ('co' in cfg.model.aux_task) + ('or' in cfg.model.aux_task) + ('fi' in cfg.model.aux_task) + ('su' in cfg.model.aux_task) + ('ma' in cfg.model.aux_task) != 0:
            assert cfg.model.aux_method in ['batch'], 'coherence & margin sub-summary task samples can not be sampled within Language_pair_dataset! '
            if cfg.model.aux_method == 'batch':
                assert cfg.model.loss_reduce is not None or cfg.model.aux_p is not None
        else:
            assert cfg.model.aux_method is None and cfg.model.loss_reduce is None
        if cfg.model.aux_task is not None:
            if 'co' in cfg.model.aux_task.split('_'):
                assert cfg.model.co_T is not None, 'coherence and margin task need T! '
            if 'ma' in cfg.model.aux_task.split('_'):
                assert cfg.model.ma_T is not None, 'coherence and margin task need T! '
                assert cfg.model.ma_sample_truncate == -1 or cfg.model.ma_sample_truncate > 0
        else:
            assert cfg.model.co_T is None, 'T param is only used by coherence and margin task! '
            assert cfg.model.ma_T is None, 'T param is only used by coherence and margin task! '
        for inspect_task, param_lst in task_param.items():
            for cur_param in param_lst:
                cur_value = getattr(cfg.model, cur_param)
                if cfg.model.aux_task is not None and inspect_task in cfg.model.aux_task.split('_'):
                    assert cur_value is not None, '{} need {} param! '.format(inspect_task, cur_param)
                    if cur_param != 'ma_version' and cur_param != 'class_lr' and 'T' not in cur_param and cur_param not in ['co_sample_truncate', 'ma_sample_truncate']:
                        assert 1 <= cur_value <= 18 if 'window' in cur_param else 30 <= cur_value <= 40, cur_param
                else:
                    assert cur_value is None, '{} param is only used by {} task! '.format(cur_param, inspect_task)
        if cfg.model.ma_version == 1:
            assert cfg.model.ma_window_size is not None
        elif cfg.model.ma_version == 2:
            assert cfg.model.ma_window_size is None
            assert cfg.model.ma2_maxwin is not None and (cfg.model.ma2_step is None or cfg.model.ma2_step is not None and cfg.model.ma2_step <= 10)
        elif cfg.model.aux_task is not None and 'ma' in cfg.model.aux_task.split('_'):
            assert False

        if cfg.model.ma2_exp != '':
            for exp_param in cfg.model.ma2_exp.split('_'):
                assert exp_param in ['allsub', 'add1', 'mutual']
    cfg.model.not_clear_cache = True if not cfg.model.data.startswith('SAMSum') else False
    cfg.model.nss = True if not cfg.model.data.startswith('SAMSum') else False
    print('cfg.model.not_clear_cache, cfg.model.not_clear_cache = %s, %s' % (cfg.model.not_clear_cache, cfg.model.nss))
    if cfg.model.class_lr is None:
        assert cfg.model.aux_task is None or 'co' not in cfg.model.aux_task
        cfg.model.class_lr = 1
    if cfg.model.ma_mask_p is not None:
        assert cfg.model.ma_mask_p != 0
        assert cfg.model.ma_lam is not None
    assert not ((cfg.model.pre_task is not None) and (cfg.model.aux_task is not None)), 'only one stage is allowed, pretraining or finetune! '
    if cfg.model.pre_task is not None:
        assert getattr(cfg.model, 'co_T', None) is None, 'only aux_task need paramter T! '
        assert getattr(cfg.model, 'ma_T', None) is None, 'only aux_task need paramter T! '

    model = task.build_model(cfg.model)

    need_origindata = (cfg.model.aux_task is not None and cfg.model.aux_method is not None or ('MLM_GSG' == cfg.model.pre_task or 'MLM_GSG' == cfg.model.aux_task))
    origindata = OriginDataset(cfg.model.data, ['train'], cfg.model.seed, cfg.model.co_window_size, cfg.model.or_window_size, cfg.model.su_window_size, cfg.model.ma_window_size,
                               cfg.model.co_truncate, cfg.model.or_truncate, cfg.model.fi_truncate, cfg.model.su_truncate, cfg.model.ma_truncate,
                               cfg.model.co_sample_truncate,
                               cfg.model.ma_sample_truncate,
                               cfg.model.aux_task, task, model, cfg.model.ma_version, cfg.model.ma2_exp,
                               cfg.model.rouge_version, cfg.model.rouge_metric, cfg.model.ma2_rm1sub,
                               ('MLM_GSG' == cfg.model.pre_task or 'MLM_GSG' == cfg.model.aux_task),
                               cfg.model.ma_rmcomp,
                               cfg.model.ma2_minwin,
                               cfg.model.ma2_maxwin,
                               cfg.model.ma2_step,
                               cfg.model.ma_shuffle,
                               cfg.model.ma_mask_p,
                               cfg.model.ma_lam,
                               cfg.model.co_input_truncate,
                               cfg.model.co_input_replace,
                               cfg.model.mediasimple) if need_origindata else None
    if need_origindata:
        assert origindata is not None
    criterion = task.build_criterion(cfg.criterion)
    # logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    logger.info('cfg.common.model_parallel_size = {}'.format(cfg.common.model_parallel_size))
    if cfg.common.model_parallel_size == 1:
        logger.info('cfg.common.model_parallel_size == 1')
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        logger.info('cfg.common.model_parallel_size != 1')
        trainer = MegatronTrainer(cfg, task, model, criterion)

    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per GPU = {} and batch size per GPU = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    logger.info('cfg.checkpoint = {}'.format(cfg.checkpoint))
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
        seed=cfg.model.seed,
        pre_task=cfg.model.pre_task,
        mask_p=cfg.model.mask_p,
        lam=cfg.model.lam,
        aux_task=cfg.model.aux_task,
        aux_p=cfg.model.aux_p,
        aux_method=cfg.model.aux_method,
        origindata=None
    )

    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    while epoch_itr.next_epoch_idx <= max_epoch:
        logger.info('before epoch {}, lr {}'.format(epoch_itr.next_epoch_idx, lr))
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break

        # train for one epoch
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr, origindata, cfg.model.new_loss_cal, cfg.model.min_oom)
        if should_stop:
            break
        #
        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train")
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr, origindata, new_loss_cal, min_oom
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))

    average_meter = AverageMeter(
        ['co_ps', 'co_ns', 'co_ps_soft', 'co_ns_soft', 'co_loss',
         'ma_l1', 'ma_l2', 'ma_l1_soft', 'ma_l2_soft', 'ma_loss'], version=1 if not cfg.model.new_loss_cal else 2)
    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()

    cur_mean = trainer.model.classification_heads.coherence_score.dense.weight.data.mean().item()
    cur_var = trainer.model.classification_heads.coherence_score.out_proj.weight.data.var().item()
    print('before training m/v {}/{}'.format(cur_mean, cur_var))
    # dw1 = trainer.model.classification_heads.coherence_score.dense.weight.data[0][:5]
    # db1 = trainer.model.classification_heads.coherence_score.dense.bias.data
    # pw1 = trainer.model.classification_heads.coherence_score.out_proj.weight.data
    # pb1 = trainer.model.classification_heads.coherence_score.out_proj.bias.data
    # print(dw1)
    # print(db1[:5])
    # print(pw1[0][:5])
    # print(pb1[:5])
    step_one_epoch = len(progress)
    print('trainer.data_parallel_world_size', trainer.data_parallel_world_size, 'step in one epoch = ', step_one_epoch)
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            if cfg.model.loss_reduce in ['alter', 'sum'] and cfg.model.aux_task.islower() and (i % 5 == 0 or i == 1):
                cur_mean = trainer.model.classification_heads.coherence_score.dense.weight.data.mean().item()
                cur_var = trainer.model.classification_heads.coherence_score.out_proj.weight.data.var().item()
                logging.error('aux_step e{} s{} {} m/v {} {}'.format(epoch_itr.epoch, i, average_meter.print(epoch_done=False), cur_mean, cur_var))
                average_meter.zero_cur()
            log_output = trainer.train_step(samples, False, origindata, i, average_meter, new_loss_cal, min_oom, epoch_itr.epoch, cfg.model.warm_aux, step_one_epoch, cfg.model.only_aux)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch, origindata, (i != 0 and cfg.model.valid_save_step is not None and i % cfg.model.valid_save_step == 0), i
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    if epoch_itr.epoch == 1:
        os.system('~/anaconda3/bin/gpustat > ./%s/gpu.log' % (cfg.checkpoint.save_dir))
        os.system('~/anaconda3/envs/fairseq/bin/gpustat >> ./%s/gpu.log' % (cfg.checkpoint.save_dir))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")

    logging.info('aux_epoch {} {}'.format(epoch_itr.epoch, average_meter.print(epoch_done=True)))
    return valid_losses, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
    origindata: None,
    valid_save_step=None,
    step_index=None
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
        cfg.optimization.stop_time_hours > 0
        and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or should_stop
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )
    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
        or should_stop
        or (
            cfg.dataset.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.dataset.validate_interval_updates == 0
        )
    ) and not cfg.dataset.disable_validation

    if valid_save_step:
        do_save = True
        do_validate = True
        logger.error('valid_save {}'.format('='*30))
    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets, origindata)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0], cfg.model.save_epoch,
        )

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    subsets: List[str],
    origindata: None
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset in subsets:
        logger.info('begin validation on "{}" subset {}'.format(subset, '=' * 66))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(shuffle=False)
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(
    cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)


if __name__ == "__main__":
    cli_main()
