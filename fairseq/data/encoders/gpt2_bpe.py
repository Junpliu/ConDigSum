# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from fairseq import file_utils
from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass

from .gpt2_bpe_utils import get_encoder


# DEFAULT_ENCODER_JSON = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json"
# DEFAULT_VOCAB_BPE = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"
DEFAULT_ENCODER_JSON = ['/export/liujunpeng/code/fairseq/train_sh/encoder.json', 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json']
DEFAULT_VOCAB_BPE = ['/export/liujunpeng/code/fairseq/train_sh/vocab.bpe', 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe']

@dataclass
class GPT2BPEConfig(FairseqDataclass):
    gpt2_encoder_json: str = field(
        default=DEFAULT_ENCODER_JSON[0], metadata={"help": "path to encoder.json"}
    )
    gpt2_vocab_bpe: str = field(
        default=DEFAULT_VOCAB_BPE[0], metadata={"help": "path to vocab.bpe"}
    )


@register_bpe("gpt2", dataclass=GPT2BPEConfig)
class GPT2BPE(object):
    def __init__(self, cfg):
        try:
            encoder_json = file_utils.cached_path(cfg.gpt2_encoder_json)
            vocab_bpe = file_utils.cached_path(cfg.gpt2_vocab_bpe)
            self.bpe = get_encoder(encoder_json, vocab_bpe)
        except:
            encoder_json = file_utils.cached_path(DEFAULT_ENCODER_JSON[1])
            vocab_bpe = file_utils.cached_path(DEFAULT_VOCAB_BPE[1])
            self.bpe = get_encoder(encoder_json, vocab_bpe)

    def encode(self, x: str) -> str:
        return " ".join(map(str, self.bpe.encode(x)))

    def decode(self, x: str) -> str:
        return self.bpe.decode(
            [int(tok) if tok not in {"<unk>", "<mask>"} else tok for tok in x.split()]
        )

    def is_beginning_of_word(self, x: str) -> bool:
        return self.decode(x).startswith(" ")
