#!/bin/bash
# Convenience wrapper around the training/evaluation entry points.
#
# Usage:
#   bash train.sh teacher                 # Stage 1: full-modality teacher
#   bash train.sh unified                 # Stage 2, unified mode (one model, 15 subsets)
#   bash train.sh separate                # Stage 2, per_subset mode (one subset)
#   bash train.sh separate_all            # Stage 2, per_subset mode (all 15 subsets)
#   bash train.sh eval_separate           # Evaluate a per_subset checkpoint
#   bash train.sh eval_unified            # Evaluate a unified checkpoint on all subsets
#
# Configurable environment variables (with defaults):
#   DATA_DIR            ./data
#   LOG_DIR             ./log
#   GPU                 0
#   MODEL_TYPE          unetr            (vnet | unetr | swin_unetr)
#   STUDENT_MODALITIES  T1               (e.g. "T1+T2"; used by 'separate')
#   UNIFIED_STRATEGY    uniform_subset   (uniform_subset | uniform_size | fixed_one_missing)
#   TEACHER_CKPT        $LOG_DIR/teacher/${MODEL_TYPE}_teacher/model/best_model.pth
#   MODEL_PATH          (required by eval_* unless the default run directory exists)

set -e

DATA_DIR="${DATA_DIR:-./data}"
LOG_DIR="${LOG_DIR:-./log}"
GPU="${GPU:-0}"
MODEL_TYPE="${MODEL_TYPE:-unetr}"
STUDENT_MODALITIES="${STUDENT_MODALITIES:-T1}"
UNIFIED_STRATEGY="${UNIFIED_STRATEGY:-uniform_subset}"
TEACHER_CKPT="${TEACHER_CKPT:-$LOG_DIR/teacher/${MODEL_TYPE}_teacher/model/best_model.pth}"

ALL_SUBSETS="T1 T2 T1ce FLAIR \
T1+T2 T1+T1ce T1+FLAIR T2+T1ce T2+FLAIR T1ce+FLAIR \
T1+T2+T1ce T1+T2+FLAIR T1+T1ce+FLAIR T2+T1ce+FLAIR \
T1+T2+T1ce+FLAIR"

require_teacher() {
    if [ ! -f "$TEACHER_CKPT" ]; then
        echo "Error: teacher checkpoint not found at $TEACHER_CKPT"
        echo "Train it first with: bash train.sh teacher"
        exit 1
    fi
}

train_separate_one() {
    local subset="$1"
    echo "[separate] model=$MODEL_TYPE subset=$subset"
    python train_prompt_distill.py \
        --training_mode per_subset \
        --model_type "$MODEL_TYPE" \
        --student_modalities "$subset" \
        --teacher_ckpt_path "$TEACHER_CKPT" \
        --freeze_stu_encoder \
        --batch_size 4 \
        --lr 0.001 \
        --max_epoch 1000 \
        --data_dir "$DATA_DIR" \
        --log_dir "$LOG_DIR/prompt_distill" \
        --gpu "$GPU"
}

case "$1" in
    teacher)
        echo "[teacher] model=$MODEL_TYPE data=$DATA_DIR"
        if [ "$MODEL_TYPE" = "vnet" ]; then
            python train_teacher.py \
                --model_type vnet \
                --batch_size 4 \
                --max_epoch 1000 \
                --lr 0.001 \
                --data_dir "$DATA_DIR" \
                --log_dir "$LOG_DIR/teacher" \
                --gpu "$GPU"
        else
            python train_teacher.py \
                --model_type "$MODEL_TYPE" \
                --pretrained \
                --batch_size 2 \
                --max_epoch 500 \
                --lr 0.0005 \
                --data_dir "$DATA_DIR" \
                --log_dir "$LOG_DIR/teacher" \
                --gpu "$GPU"
        fi
        echo "Best teacher checkpoint: $LOG_DIR/teacher/${MODEL_TYPE}_teacher/model/best_model.pth"
        ;;

    unified)
        require_teacher
        echo "[unified] model=$MODEL_TYPE strategy=$UNIFIED_STRATEGY"
        python train_prompt_distill.py \
            --training_mode unified \
            --model_type "$MODEL_TYPE" \
            --unified_sample_strategy "$UNIFIED_STRATEGY" \
            --unified_val_subsets default \
            --student_modalities "$STUDENT_MODALITIES" \
            --teacher_ckpt_path "$TEACHER_CKPT" \
            --freeze_stu_encoder \
            --batch_size 4 \
            --lr 0.001 \
            --max_epoch 1000 \
            --data_dir "$DATA_DIR" \
            --log_dir "$LOG_DIR/prompt_distill" \
            --gpu "$GPU"
        ;;

    separate)
        require_teacher
        train_separate_one "$STUDENT_MODALITIES"
        ;;

    separate_all)
        require_teacher
        for subset in $ALL_SUBSETS; do
            train_separate_one "$subset"
        done
        ;;

    eval_separate)
        MODEL_PATH="${MODEL_PATH:-$LOG_DIR/prompt_distill/${MODEL_TYPE}_${STUDENT_MODALITIES}/model/best_model.pth}"
        if [ ! -f "$MODEL_PATH" ]; then
            echo "Error: checkpoint not found at $MODEL_PATH (set MODEL_PATH explicitly)"
            exit 1
        fi
        python evaluate_prompt.py \
            --model_path "$MODEL_PATH" \
            --data_dir "$DATA_DIR" \
            --output_path "./results/separate_${MODEL_TYPE}_${STUDENT_MODALITIES}" \
            --gpu "$GPU"
        ;;

    eval_unified)
        MODEL_PATH="${MODEL_PATH:-$LOG_DIR/prompt_distill/${MODEL_TYPE}_unified_${UNIFIED_STRATEGY}/model/best_model.pth}"
        if [ ! -f "$MODEL_PATH" ]; then
            echo "Error: checkpoint not found at $MODEL_PATH (set MODEL_PATH explicitly)"
            exit 1
        fi
        python evaluate_unified.py \
            --model_path "$MODEL_PATH" \
            --data_dir "$DATA_DIR" \
            --subsets all \
            --output_path "./results/unified_${MODEL_TYPE}_${UNIFIED_STRATEGY}" \
            --gpu "$GPU"
        ;;

    *)
        echo "Usage: bash train.sh {teacher|unified|separate|separate_all|eval_separate|eval_unified}"
        echo "See the header of this script (or README.md) for configurable environment variables."
        exit 1
        ;;
esac

echo "Done."
