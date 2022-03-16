#!/usr/bin/env bash
source ~/anaconda3/bin/activate fairseq

# SAMSumInd
#TASK='SAMSumInd'  # dataset name
#for SPLIT in train val test
#do
#    python -m examples.roberta.multiprocessing_bpe_encoder \
#    --encoder-json encoder.json \
#    --vocab-bpe vocab.bpe \
#    --inputs "$TASK/$SPLIT.json" \
#    --outputs "$TASK/$SPLIT.bpe.source" \
#    --workers 60 \
#    --keep-empty \
#    --encode_ind \
#    --sep_id_ind 39811 \
#    --dataset 'SAMSumInd';
#    python -m examples.roberta.multiprocessing_bpe_encoder \
#    --encoder-json encoder.json \
#    --vocab-bpe vocab.bpe \
#    --inputs "$TASK/$SPLIT.target" \
#    --outputs "$TASK/$SPLIT.bpe.target" \
#    --workers 60 \
#    --keep-empty;
#done

# mediasum
#TASK='mediasum'  # dataset name
#for SPLIT in val test train
#do
#    python -m examples.roberta.multiprocessing_bpe_encoder \
#    --encoder-json encoder.json \
#    --vocab-bpe vocab.bpe \
#    --inputs "$TASK/${SPLIT}_dialog.json" \
#    --outputs "$TASK/$SPLIT.bpe.source" \
#    --workers 60 \
#    --keep-empty \
#    --encode_ind \
#    --sep_id_ind 39811 \
#    --truncate_size 1021 \
#    --dataset 'mediasum';
#    python -m examples.roberta.multiprocessing_bpe_encoder \
#    --encoder-json encoder.json \
#    --vocab-bpe vocab.bpe \
#    --inputs "$TASK/$SPLIT.target" \
#    --outputs "$TASK/$SPLIT.bpe.target" \
#    --workers 60 \
#    --keep-empty;
#done


TASK='customdata'  # dataset name
for SPLIT in train val test
do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$TASK/$SPLIT.jsonl" \
    --outputs "$TASK/$SPLIT.bpe.source" \
    --workers 60 \
    --keep-empty \
    --encode_ind \
    --sep_id_ind 39811 \
    --dataset 'customdata';
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$TASK/$SPLIT.target" \
    --outputs "$TASK/$SPLIT.bpe.target" \
    --workers 60 \
    --keep-empty;
done