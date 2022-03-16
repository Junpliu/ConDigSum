if [ $# != 2 ] ; then
echo "USAGE: $0 COMMENT"
echo " e.g.: $0 condigsum 2"
exit 1;
fi

GPU=${2}
if [ ${GPU} -eq -1 ] ; then
  GPU=0,1,2,3
fi
echo "running on GPU ${GPU}"

WARMUP_UPDATES=200
MAX_TOKENS=800
UPDATE_FREQ=32

LR=4e-05
TOTAL_NUM_UPDATES=1000
dataset='SAMSumInd'
PRETRAIN_PATH='/export/liujunpeng/pretrained/bart/bart.large/model.pt'  # [YOUR_OWN_PATH]  # to be updated....

CHECKPOINT_PATH="checkpoint_${1}_$dataset" && mkdir -p $CHECKPOINT_PATH
CUDA_VISIBLE_DEVICES=${GPU} python ../fairseq_cli/train.py $dataset \
    --restore-file $PRETRAIN_PATH \
    --max-tokens $MAX_TOKENS \
    --no-last-checkpoints \
    --task translation \
    --co_window_size 14 \
    --ma2_minwin 5 \
    --ma2_maxwin 24 \
    --ma2_step 1 \
    --co_sample_truncate 2 \
    --ma_sample_truncate 2 \
    --co_loss_weight 0.005 \
    --ma_loss_weight 0.0001 \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --ddp-backend=no_c10d \
    --required-batch-size-multiple 1 \
    --no-epoch-checkpoints \
    --save-dir $CHECKPOINT_PATH \
    --seed 14632 >> ${CHECKPOINT_PATH}/log
