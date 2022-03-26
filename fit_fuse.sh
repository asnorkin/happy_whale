export CUDA_VISIBLE_DEVICES=0,
export OPENBLAS_NUM_THREADS=1

source ../.whale_venv/bin/activate
which python

WORK_DIR=fuse
LABELS_CSV=../data/train.csv
IMAGE_IDS_FILE=../data/image_ids.npy
EMBEDDINGS_FILE=../data/tf_effb7_ns_m0.4_yolov5_9c_v3_augv3_2ldp0.4_512x512_4g20x3b60e0.001lr_embeddings.pt

BATCH=256
ACC=1
GPUS=1
EPOCHS=10
WORKERS=16
FOLD=0
LR=0.001
M=0.4
DROP_FIN=0.5

EXPERIMENT=df${DROP_FIN}_${GPUS}g${BATCH}x${ACC}b${EPOCHS}e${LR}lr${FOLD}f

export PYTHONPATH=.:${PYTHONPATH}

nohup python ${WORK_DIR}/train.py \
    --precision=16 \
    --drop_fin_prob=${DROP_FIN} \
    --m=${M} \
    --labels_csv=${LABELS_CSV} \
    --image_ids_file=${IMAGE_IDS_FILE} \
    --embeddings_file=${EMBEDDINGS_FILE} \
    --gpus=${GPUS} \
    --batch_size=${BATCH} \
    --accumulate_grad_batches=${ACC} \
    --num_epochs=${EPOCHS} \
    --num_workers=${WORKERS} \
    --lr=${LR} \
    --fold=${FOLD} \
    --experiment=${EXPERIMENT} > ${WORK_DIR}/logs/${EXPERIMENT}_train.log 2>&1 &
