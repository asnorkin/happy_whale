export CUDA_VISIBLE_DEVICES=0,1,2,3,
export OPENBLAS_NUM_THREADS=1

source ../.whale_venv/bin/activate
which python

WORK_DIR=pipeline
LABELS_CSV=../data/train.csv
IMAGES_DIR=../data/train_images_yolov5_9c_v3_512

B=0
BATCH=60
ACC=1
GPUS=4
EPOCHS=60
WORKERS=16
INPUT_WIDTH=512
INPUT_HEIGHT=512
FOLD=0
DROPOUT=0.4
LR=0.001
M=0.4

EXPERIMENT=tf_effb${B}_ns_m${M}_yolov5_9c_v3_augv3_2ldp${DROPOUT}_${INPUT_WIDTH}x${INPUT_HEIGHT}_${GPUS}g${BATCH}x${ACC}b${EPOCHS}e${LR}lr${FOLD}f
MODEL=tf_efficientnet_b${B}_ns

export PYTHONPATH=.:${PYTHONPATH}

nohup python ${WORK_DIR}/train.py \
    --random_image=1 \
    --precision=16 \
    --m=${M} \
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

