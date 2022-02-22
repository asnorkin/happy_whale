source ../.whale_venv/bin/activate
which python

WORK_DIR=pipeline
LABELS_CSV=../data/train.csv
EXPERIMENT=baseline

BATCH=32
EPOCHS=10
WORKERS=16

export PYTHONPATH=.:${PYTHONPATH}

nohup python ${WORK_DIR}/train.py \
    --labels_csv=${LABELS_CSV} \
    --gpus=1 \
    --batch_size=${BATCH} \
    --num_epochs=${EPOCHS} \
    --num_workers=${WORKERS} \
    --experiment=${EXPERIMENT} > ${WORK_DIR}/logs/${EXPERIMENT}_train.log 2>&1 &

