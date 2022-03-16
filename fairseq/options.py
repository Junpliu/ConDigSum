# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Callable, List, Optional

import torch
from fairseq import utils
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    EvalLMConfig,
    GenerationConfig,
    InteractiveConfig,
    OptimizationConfig,
)
from fairseq.dataclass.utils import gen_parser_from_dataclass

# this import is for backward compatibility
from fairseq.utils import csv_str_list, eval_bool, eval_str_dict, eval_str_list  # noqa


def get_preprocessing_parser(default_task="translation"):
    parser = get_parser("Preprocessing", default_task)
    add_preprocess_args(parser)
    return parser


def get_training_parser(default_task="translation"):
    parser = get_parser("Trainer", default_task)
    add_dataset_args(parser, train=True)
    add_distributed_training_args(parser)
    add_model_args(parser)
    add_optimization_args(parser)
    add_checkpoint_args(parser)
    return parser


def get_generation_parser(interactive=False, default_task="translation"):
    parser = get_parser("Generation", default_task)
    add_dataset_args(parser, gen=True)
    add_distributed_training_args(parser, default_world_size=1)
    add_generation_args(parser)
    add_checkpoint_args(parser)
    if interactive:
        add_interactive_args(parser)
    return parser


def get_interactive_generation_parser(default_task="translation"):
    return get_generation_parser(interactive=True, default_task=default_task)


def get_eval_lm_parser(default_task="language_modeling"):
    parser = get_parser("Evaluate Language Model", default_task)
    add_dataset_args(parser, gen=True)
    add_distributed_training_args(parser, default_world_size=1)
    add_eval_lm_args(parser)
    return parser


def get_validation_parser(default_task=None):
    parser = get_parser("Validation", default_task)
    add_dataset_args(parser, train=True)
    add_distributed_training_args(parser, default_world_size=1)
    group = parser.add_argument_group("Evaluation")
    gen_parser_from_dataclass(group, CommonEvalConfig())
    return parser


def parse_args_and_arch(
    parser: argparse.ArgumentParser,
    input_args: List[str] = None,
    parse_known: bool = False,
    suppress_defaults: bool = False,
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None,
):
    """
    Args:
        parser (ArgumentParser): the parser
        input_args (List[str]): strings to parse, defaults to sys.argv
        parse_known (bool): only parse known arguments, similar to
            `ArgumentParser.parse_known_args`
        suppress_defaults (bool): parse while ignoring all default values
        modify_parser (Optional[Callable[[ArgumentParser], None]]):
            function to modify the parser, e.g., to set default values
    """
    if suppress_defaults:
        # Parse args without any default values. This requires us to parse
        # twice, once to identify all the necessary task/model args, and a second
        # time with all defaults set to None.
        args = parse_args_and_arch(
            parser,
            input_args=input_args,
            parse_known=parse_known,
            suppress_defaults=False,
        )
        suppressed_parser = argparse.ArgumentParser(add_help=False, parents=[parser])
        suppressed_parser.set_defaults(**{k: None for k, v in vars(args).items()})
        args = suppressed_parser.parse_args(input_args)
        return argparse.Namespace(
            **{k: v for k, v in vars(args).items() if v is not None}
        )

    from fairseq.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY, MODEL_REGISTRY

    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    usr_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    usr_parser.add_argument("--user-dir", default=None)
    usr_args, _ = usr_parser.parse_known_args(input_args)
    utils.import_user_module(usr_args)

    if modify_parser is not None:
        modify_parser(parser)

    # The parser doesn't know about model/criterion/optimizer-specific args, so
    # we parse twice. First we parse the model/criterion/optimizer, then we
    # parse a second time after adding the *-specific arguments.
    # If input_args is given, we will parse those args instead of sys.argv.
    args, _ = parser.parse_known_args(input_args)

    # Add model-specific args to parser.
    if hasattr(args, "arch"):
        model_specific_group = parser.add_argument_group(
            "Model-specific configuration",
            # Only include attributes which are explicitly given as command-line
            # arguments or which have default values.
            argument_default=argparse.SUPPRESS,
        )
        if args.arch in ARCH_MODEL_REGISTRY:
            ARCH_MODEL_REGISTRY[args.arch].add_args(model_specific_group)
        elif args.arch in MODEL_REGISTRY:
            MODEL_REGISTRY[args.arch].add_args(model_specific_group)
        else:
            raise RuntimeError()

    if hasattr(args, "task"):
        from fairseq.tasks import TASK_REGISTRY
        TASK_REGISTRY[args.task].add_args(parser)

    if hasattr(args, "aux_task"):
        print('running script, args has aux_task', args.aux_task)
    else:
        print('=' * 100)

    if getattr(args, "use_bmuf", False):
        # hack to support extra args for block distributed data parallelism
        from fairseq.optim.bmuf import FairseqBMUF

        FairseqBMUF.add_args(parser)

    # Add *-specific args to parser.
    from fairseq.registry import REGISTRIES

    for registry_name, REGISTRY in REGISTRIES.items():
        choice = getattr(args, registry_name, None)
        if choice is not None:
            cls = REGISTRY["registry"][choice]
            if hasattr(cls, "add_args"):
                cls.add_args(parser)
            elif hasattr(cls, "__dataclass"):
                gen_parser_from_dataclass(parser, cls.__dataclass())

    # Modify the parser a second time, since defaults may have been reset
    if modify_parser is not None:
        modify_parser(parser)

    # Parse a second time.
    if parse_known:
        args, extra = parser.parse_known_args(input_args)
    else:
        args = parser.parse_args(input_args)
        extra = None
    # Post-process args.
    if (
        hasattr(args, "batch_size_valid") and args.batch_size_valid is None
    ) or not hasattr(args, "batch_size_valid"):
        args.batch_size_valid = args.batch_size
    if hasattr(args, "max_tokens_valid") and args.max_tokens_valid is None:
        args.max_tokens_valid = args.max_tokens
    if getattr(args, "memory_efficient_fp16", False):
        args.fp16 = True
    if getattr(args, "memory_efficient_bf16", False):
        args.bf16 = True
    args.tpu = getattr(args, "tpu", False)
    args.bf16 = getattr(args, "bf16", False)
    if args.bf16:
        args.tpu = True
    if args.tpu and args.fp16:
        raise ValueError("Cannot combine --fp16 and --tpu, use --bf16 on TPUs")

    if getattr(args, "seed", None) is None:
        assert False, 'no seed'
        args.seed = 1  # default seed for training
        args.no_seed_provided = True
    else:
        args.no_seed_provided = False

    # Apply architecture configuration.
    if hasattr(args, "arch") and args.arch in ARCH_CONFIG_REGISTRY:
        ARCH_CONFIG_REGISTRY[args.arch](args)

    if parse_known:
        return args, extra
    else:
        return args


def get_parser(desc, default_task="translation"):
    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    usr_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    usr_parser.add_argument("--user-dir", default=None)
    usr_args, _ = usr_parser.parse_known_args()
    utils.import_user_module(usr_args)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    gen_parser_from_dataclass(parser, CommonConfig())

    from fairseq.registry import REGISTRIES

    for registry_name, REGISTRY in REGISTRIES.items():
        parser.add_argument(
            "--" + registry_name.replace("_", "-"),
            default=REGISTRY["default"],
            choices=REGISTRY["registry"].keys(),
        )

    # Task definitions can be found under fairseq/tasks/
    from fairseq.tasks import TASK_REGISTRY

    parser.add_argument(
        "--task",
        metavar="TASK",
        default=default_task,
        choices=TASK_REGISTRY.keys(),
        help="task",
    )
    # fmt: on
    return parser


def add_preprocess_args(parser):
    group = parser.add_argument_group("Preprocessing")
    # fmt: off
    group.add_argument("-s", "--source-lang", default=None, metavar="SRC",
                       help="source language")
    group.add_argument("-t", "--target-lang", default=None, metavar="TARGET",
                       help="target language")
    group.add_argument("--trainpref", metavar="FP", default=None,
                       help="train file prefix (also used to build dictionaries)")
    group.add_argument("--validpref", metavar="FP", default=None,
                       help="comma separated, valid file prefixes "
                            "(words missing from train set are replaced with <unk>)")
    group.add_argument("--testpref", metavar="FP", default=None,
                       help="comma separated, test file prefixes "
                            "(words missing from train set are replaced with <unk>)")
    group.add_argument("--align-suffix", metavar="FP", default=None,
                       help="alignment file suffix")
    group.add_argument("--destdir", metavar="DIR", default="data-bin",
                       help="destination dir")
    group.add_argument("--thresholdtgt", metavar="N", default=0, type=int,
                       help="map words appearing less than threshold times to unknown")
    group.add_argument("--thresholdsrc", metavar="N", default=0, type=int,
                       help="map words appearing less than threshold times to unknown")
    group.add_argument("--tgtdict", metavar="FP",
                       help="reuse given target dictionary")
    group.add_argument("--srcdict", metavar="FP",
                       help="reuse given source dictionary")
    group.add_argument("--nwordstgt", metavar="N", default=-1, type=int,
                       help="number of target words to retain")
    group.add_argument("--nwordssrc", metavar="N", default=-1, type=int,
                       help="number of source words to retain")
    group.add_argument("--alignfile", metavar="ALIGN", default=None,
                       help="an alignment file (optional)")
    parser.add_argument('--dataset-impl', metavar='FORMAT', default='mmap',
                        choices=get_available_dataset_impl(),
                        help='output dataset implementation')
    group.add_argument("--joined-dictionary", action="store_true",
                       help="Generate joined dictionary")
    group.add_argument("--only-source", action="store_true",
                       help="Only process the source language")
    group.add_argument("--padding-factor", metavar="N", default=8, type=int,
                       help="Pad dictionary size to be multiple of N")
    group.add_argument("--workers", metavar="N", default=1, type=int,
                       help="number of parallel workers")
    # fmt: on
    return parser


def add_dataset_args(parser, train=False, gen=False):
    group = parser.add_argument_group("dataset_data_loading")
    gen_parser_from_dataclass(group, DatasetConfig())
    # fmt: on
    return group


def add_distributed_training_args(parser, default_world_size=None):
    group = parser.add_argument_group("distributed_training")
    if default_world_size is None:
        default_world_size = max(1, torch.cuda.device_count())
    gen_parser_from_dataclass(
        group, DistributedTrainingConfig(distributed_world_size=default_world_size)
    )
    return group


def add_optimization_args(parser):
    group = parser.add_argument_group("optimization")
    # fmt: off
    gen_parser_from_dataclass(group, OptimizationConfig())
    # fmt: on
    return group


def add_checkpoint_args(parser):
    group = parser.add_argument_group("checkpoint")
    # fmt: off
    gen_parser_from_dataclass(group, CheckpointConfig())
    # fmt: on
    return group


def add_common_eval_args(group):
    gen_parser_from_dataclass(group, CommonEvalConfig())


def add_eval_lm_args(parser):
    group = parser.add_argument_group("LM Evaluation")
    add_common_eval_args(group)
    gen_parser_from_dataclass(group, EvalLMConfig())


def add_generation_args(parser):
    group = parser.add_argument_group("Generation")
    add_common_eval_args(group)
    gen_parser_from_dataclass(group, GenerationConfig())
    return group


def add_interactive_args(parser):
    group = parser.add_argument_group("Interactive")
    gen_parser_from_dataclass(group, InteractiveConfig())


def add_model_args(parser):
    from typing import Union
    group = parser.add_argument_group("Model configuration")
    # fmt: off

    # Model definitions can be found under fairseq/models/
    #
    # The model architecture can be specified in several ways.
    # In increasing order of priority:
    # 1) model defaults (lowest priority)
    # 2) --arch argument
    # 3) --encoder/decoder-* arguments (highest priority)
    from fairseq.models import ARCH_MODEL_REGISTRY
    group.add_argument('--arch', '-a', metavar='ARCH',
                       choices=ARCH_MODEL_REGISTRY.keys(),
                       help='model architecture')
    group.add_argument('--pre_task',
                       default=None,
                       type=str,
                       help='pretrain task, SR(sentence reordering), TI(text infilling). ')
    group.add_argument('--mask_p',
                       default=None,
                       type=float,
                       help='pretrain task, SR(sentence reordering), TI(text infilling). ')
    group.add_argument('--lam',
                       default=None,
                       type=int,
                       help='pretrain task, SR(sentence reordering), TI(text infilling). ')

    group.add_argument('--aux_task',
                       default='co_ma',
                       type=str,
                       help='auxiliary task, (coherence / margin_sub_summary)')
    group.add_argument('--aux_p',
                       default=None,
                       type=float,
                       help='auxiliary task, (coherence / margin_sub_summary)')
    group.add_argument('--aux_method',
                       default='batch',
                       type=str,
                       help='auxiliary task, (coherence / margin_sub_summary)')

    group.add_argument('--rouge_version',
                       default='2',
                       type=str,
                       help='rouge_version')
    group.add_argument('--rouge_metric',
                       default='r',
                       type=str,
                       help='rouge_metric: p/r/f')
    group.add_argument('--ma2_rm1sub',
                       default=True,
                       help='ma2_rm1sub')
    group.add_argument('--ma2_minwin',
                       default=1,
                       type=int,
                       help='min window size of margin sub-summary task')
    group.add_argument('--ma2_maxwin',
                       default=5,
                       type=int,
                       help='max window size of margin sub-summary task')
    group.add_argument('--ma2_step',
                       default=None,
                       type=int,
                       help='ma2, the step size of searching windows. ')
    group.add_argument('--ma_version',
                       default=2,
                       type=int,
                       help='ma version(v1/v2) ')
    group.add_argument('--co_sample_truncate',
                       default=None,
                       type=int,
                       help='co sample truncate, for the balance of training samples\'s number between coherence and other tasks. ')
    group.add_argument('--ma_sample_truncate',
                       default=None,
                       type=int,
                       help='ma sample truncate, for the balance of training samples\'s number between sub-summary task and other tasks. ')
    group.add_argument('--co_window_size',
                       default=None,
                       type=int,
                       help='windows size for creating negtive samples')
    group.add_argument('--co_truncate',
                       default=40,
                       type=int,
                       help='truncated size for creating negtive samples')
    group.add_argument('--or_truncate',
                       default=None,
                       type=int,
                       help='truncated size for creating negtive samples')
    group.add_argument('--fi_truncate',
                       default=None,
                       type=int,
                       help='truncated size for creating negtive samples')
    group.add_argument('--su_truncate',
                       default=None,
                       type=int,
                       help='truncated size for creating negtive samples')
    group.add_argument('--ma_truncate',
                       default=40,
                       type=int,
                       help='truncated size for creating negtive samples')
    group.add_argument('--or_window_size',
                       default=None,
                       type=int,
                       help='windows size for creating negtive samples(order task)')
    group.add_argument('--su_window_size',
                       default=None,
                       type=int,
                       help='windows size for creating negtive samples(sub-summrization task)')
    group.add_argument('--ma_window_size',
                       default=None,
                       type=int,
                       help='windows size for creating negtive samples(margin sub-summarization task)')
    group.add_argument('--co_T',
                       default=1.0,
                       type=float,
                       help='temparature')
    group.add_argument('--ma_T',
                       default=1.0,
                       type=float,
                       help='temparature')
    group.add_argument('--ma_rmcomp',
                       default=False,
                       action='store_true',
                       help='remove contrastive loss. ')
    group.add_argument('--ma_shuffle',
                       default=False,
                       action='store_true',
                       help='shuffle the input of sub-summary task. ')
    group.add_argument('--ma_mask_p',
                       default=None,
                       type=float,
                       help='mask the tokens of input source. ')
    group.add_argument('--ma_lam',
                       default=None,
                       type=int,
                       help='the lambda of poisson distribution in the sub-summary task. ')
    group.add_argument('--class_lr',
                       default=1.0,
                       type=float)
    group.add_argument('--new_loss_cal',
                       default=True,
                       help='new loss calculation. ')
    group.add_argument('--loss_reduce',
                       default='alter',
                       type=str,
                       help='loss backward method (alter/sum)')

    group.add_argument('--co_loss_weight',
                       default=1.0,
                       type=float)
    group.add_argument('--or_loss_weight',
                       default=1.0,
                       type=float)
    group.add_argument('--fi_loss_weight',
                       default=1.0,
                       type=float)
    group.add_argument('--su_loss_weight',
                       default=1.0,
                       type=float)
    group.add_argument('--ma_loss_weight',
                       default=1.0,
                       type=float)

    group.add_argument('--debug_forward',
                       default=False,
                       action='store_true',
                       help='debug forward only, do not backward')

    group.add_argument('--ma2_exp',
                       default='',
                       type=str,
                       help='ma2 parameters in experiments, [allsub]_[add1]. ')

    group.add_argument('--save_epoch',
                       default='',
                       type=str,
                       help='additional ckpt saved. ')
    group.add_argument('--min_oom',
                       action='store_true',
                       default=False,
                       help='minimumizing oom error! ')
    group.add_argument('--conti_some',
                       default=True,
                       help='continue operation only at the end of epochs! ')
    group.add_argument('--warm_aux',
                       action='store_true',
                       default=False,
                       help='warm aux_task! ')
    group.add_argument('--only_aux',
                       action='store_true',
                       default=False,
                       help='only aux_task')
    group.add_argument('--co_input_truncate',
                       type=int,
                       default=None,
                       help='co_input_truncate')
    group.add_argument('--co_input_replace',
                       action='store_true',
                       default=False,
                       help='co_input_replace')
    group.add_argument('--mediasimple',
                       action='store_true',
                       default=False,
                       help='mediasimple')
    group.add_argument('--not_clear_cache',
                       action='store_true',
                       default=False,
                       help='not_clear_cache')
    group.add_argument('--nss',
                       action='store_true',
                       default=False,
                       help='new calculation method for sample size')
    group.add_argument('--noaux',
                       action='store_true',
                       default=False,
                       help='strictly do not carry out auxiliary tasks. ')
    group.add_argument('--margin_delta_co',
                       type=float,
                       default=1.0,
                       help='margin coefficients for auxiliary objectives. ')
    group.add_argument('--margin_delta_su',
                       type=float,
                       default=1.0,
                       help='margin coefficients for auxiliary objectives. ')
    group.add_argument('--valid_save_step',
                       type=int,
                       default=None,
                       help='saving model every n steps. ')
    # fmt: on
    return group
