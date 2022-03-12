export CUDA_VISIBLE_DEVICES=0,

source ../.whale_venv/bin/activate
which python

WORK_DIR=camera_viewpoint
LABELS_CSV=../data/camera_viewpoint_labels.csv
IMAGES_DIR=../data/train_images_yolov5_fin_fish_v1_512

B=0
BATCH=64
ACC=1
EPOCHS=20
WORKERS=16
INPUT_WIDTH=224
INPUT_HEIGHT=224
DROPOUT=0.5
LR=0.001

EXPERIMENT=tf_effb${B}_ns_fin_fish_v1_${DROPOUT}dp_${INPUT_WIDTH}x${INPUT_HEIGHT}_${BATCH}x${ACC}b${EPOCHS}e${LR}lr
MODEL=tf_efficientnet_b${B}_ns

export PYTHONPATH=.:${PYTHONPATH}

nohup python ${WORK_DIR}/train.py \
    --model_name=${MODEL} \
    --labels_csv=${LABELS_CSV} \
    --images_dir=${IMAGES_DIR} \
    --batch_size=${BATCH} \
    --accumulate_grad_batches=${ACC} \
    --num_epochs=${EPOCHS} \
    --num_workers=${WORKERS} \
    --input_height=${INPUT_HEIGHT} \
    --input_width=${INPUT_WIDTH} \
    --dropout=${DROPOUT} \
    --lr=${LR} \
    --experiment=${EXPERIMENT} > ${WORK_DIR}/logs/${EXPERIMENT}_train.log 2>&1 &

