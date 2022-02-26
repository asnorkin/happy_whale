source ../.whale_venv/bin/activate
which python

WORK_DIR=pipeline
LABELS_CSV=../data/train.csv
IMAGES_DIR=../data/train_images_detic_512

BATCH=28
ACC=4
EPOCHS=20
WORKERS=16
INPUT_SIZE=448
LR=0.0001

EXPERIMENT=tf_effb0_ns_detic_${INPUT_SIZE}inp${BATCH}x${ACC}b${EPOCHS}e${LR}lr

export PYTHONPATH=.:${PYTHONPATH}

nohup python ${WORK_DIR}/train.py \
    --labels_csv=${LABELS_CSV} \
    --images_dir=${IMAGES_DIR} \
    --gpus=1 \
    --batch_size=${BATCH} \
    --accumulate_grad_batches=${ACC} \
    --num_epochs=${EPOCHS} \
    --num_workers=${WORKERS} \
    --input_size=${INPUT_SIZE} \
    --lr=${LR} \
    --experiment=${EXPERIMENT} > ${WORK_DIR}/logs/${EXPERIMENT}_train.log 2>&1 &

