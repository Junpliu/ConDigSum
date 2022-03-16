#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import sys
from collections import Counter
from multiprocessing import Pool
import json

from fairseq.data.encoders.gpt2_bpe import get_encoder


def main():
    """
    Helper script to encode raw text with the GPT-2 BPE using multiple processes.

    The encoder.json and vocab.bpe files can be obtained here:
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-json",
        help="path to encoder.json",
    )
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        help="path to vocab.bpe",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["-"],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=["-"],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument('--encode_ind', action='store_true', default=False)
    parser.add_argument('--sep_id_ind', type=int, default=None)
    parser.add_argument('--truncate_size', type=int, default=None)
    parser.add_argument('--dataset', default=None)
    args = parser.parse_args()

    import os
    assert os.path.exists(args.encoder_json) and os.path.exists(args.vocab_bpe)
    assert len(args.inputs) == len(
        args.outputs
    ), "number of input and output paths should match"
    if args.encode_ind:
        assert args.sep_id_ind is not None

    with contextlib.ExitStack() as stack:
        if not args.encode_ind:  # target bpe, all dataset
            inputs = [
                stack.enter_context(open(input, "r", encoding="utf-8"))
                if input != "-"
                else sys.stdin
                for input in args.inputs
            ]
        elif args.dataset is None:  # source bpe, SAMSum dataset
            with open(args.inputs[0], 'r', encoding='utf-8') as f:
                dialog_lst = json.load(f)
            input_lst = []
            for dialog in dialog_lst:
                sentences = dialog['dialogue'].split('\r\n')
                if len(dialog['dialogue'].split('\r\n')) <= 1:
                    sentences = dialog['dialogue'].split('\n')
                if len(sentences) == 1 and len(input_lst) in [14731, 818, 819]:
                    continue
                sentences = [sent.strip() for sent in sentences if len(sent.strip()) > 0]
                input_lst.append(sentences)
            inputs = [input_lst]
        elif args.dataset.startswith('mediasum'):  # source bpe, mediasum
            print('mediasum dataset, reading json file... ')
            with open(args.inputs[0], 'r', encoding='ascii') as f:
                dialog_lst = json.load(f)
            print('reading json done! ')
            input_lst = []
            for dialog in dialog_lst:
                sentences = []
                word_count = 0
                for speaker, utter in zip(dialog['speaker'], dialog['utt']):
                    temp = '%s: %s' % (speaker, utter)
                    word_count += len(temp.split())
                    sentences.append(temp)
                    if args.truncate_size is not None:
                        if word_count > args.truncate_size + 50:
                            break
                input_lst.append(sentences)
            print('len(input_lst)', len(input_lst))
            inputs = [input_lst]
        else:  # source bpe, custom dataset
            print('custom dataset! ')
            with open(args.inputs[0], 'r', encoding='utf-8') as f:
                json_lines = f.readlines()

            input_lst = []
            target_lines = []
            for line in json_lines:
                dialog = json.loads(line)
                sentences = dialog['utterance']
                input_lst.append(sentences)
                target_lines.append(dialog['summary'])
            with open(args.inputs[0].replace('jsonl', 'target'), 'w', encoding='utf-8') as g:
                g.write('\n'.join(target_lines))
            print('len(input_lst)', len(input_lst))
            inputs = [input_lst]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-"
            else sys.stdout
            for output in args.outputs
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        if not args.encode_ind:
            encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)
        else:
            encoded_lines = pool.imap(encoder.encode_lines_ind, zip(*inputs), 100)

        stats = Counter()
        for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(enc_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


class MultiprocessingEncoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            tokens = self.encode(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def encode_lines_ind(self, lines):
        if self.args.dataset.startswith('mediasum'):
            assert self.args.truncate_size is not None
        enc_lines = []
        for line in lines:
            line = [l.strip() for l in line]
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            line_lst = []
            for l in line:
                tokens = self.encode(l)
                line_lst.append(' '.join(tokens))
            if self.args.truncate_size is None:
                enc_lines.append((' %s ' % self.args.sep_id_ind).join(line_lst))
            else:
                temp = (' %s ' % self.args.sep_id_ind).join(line_lst).split()[:self.args.truncate_size]
                enc_lines.append(' '.join(temp))
        return ["PASS", enc_lines]


    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


if __name__ == "__main__":
    main()
