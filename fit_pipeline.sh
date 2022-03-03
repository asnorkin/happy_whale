export CUDA_VISIBLE_DEVICES=2,3,
export OPENBLAS_NUM_THREADS=1

source ../.whale_venv/bin/activate
which python

WORK_DIR=pipeline
LABELS_CSV=../data/train.csv
IMAGES_DIR=../data/train_images_detic_768x384

B=0
BATCH=57
ACC=1
GPUS=2
EPOCHS=20
WORKERS=16
INPUT_WIDTH=768
INPUT_HEIGHT=384
DROPOUT=0.2
LR=0.001

EXPERIMENT=tf_effb${B}_ns_detic_2ldp0.2_${INPUT_WIDTH}x${INPUT_HEIGHT}_${GPUS}g${BATCH}x${ACC}b${EPOCHS}e${LR}lr
MODEL=tf_efficientnet_b${B}_ns

export PYTHONPATH=.:${PYTHONPATH}

nohup python ${WORK_DIR}/train.py \
    --m=0.3 \
    --model_name=${MODEL} \
    --labels_csv=${LABELS_CSV} \
    --images_dir=${IMAGES_DIR} \
    --gpus=${GPUS} \
    --strategy=ddp \
    --batch_size=${BATCH} \
    --accumulate_grad_batches=${ACC} \
    --num_epochs=${EPOCHS} \
    --num_workers=${WORKERS} \
    --input_height=${INPUT_HEIGHT} \
    --input_width=${INPUT_WIDTH} \
    --dropout=${DROPOUT} \
    --lr=${LR} \
    --experiment=${EXPERIMENT} > ${WORK_DIR}/logs/${EXPERIMENT}_train.log 2>&1 &

