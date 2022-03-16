#!/usr/bin/env bash
source ~/anaconda3/bin/activate fairseq

TASK='customdata'  # TASK='SAMSumInd'  # TASK='mediasum'
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "$TASK/train.bpe" \
  --validpref "$TASK/val.bpe" \
  --destdir "$TASK/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
