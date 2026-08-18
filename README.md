# DeDP-KD: Decoupled Dual Prompting and KnowledgeDistillation for Medical Segmentation with Missing Modality

Official implementation of prompt-based knowledge distillation for BraTS brain tumor segmentation with missing modalities. A full-modality teacher transfers knowledge to a prompt-conditioned student that operates on arbitrary subsets of the four MRI modalities (T1, T2, T1ce, FLAIR).

## Highlights

- **Two training paradigms**
  - **Unified training** (`--training_mode unified`): a single student handles all 15 non-empty modality subsets. Each batch randomly samples a subset, missing channels are zero-filled, and all expert prompt pools are optimized jointly.
  - **Separate training** (`--training_mode per_subset`): one specialist student per fixed modality subset, following the classical missing-modality distillation setup.
- **Dual-prompt design**: shared *general* prompts capture modality-agnostic knowledge, while per-subset *expert* prompts encode subset-specific adaptation.
- **Multiple backbones**: VNet, UNETR, and Swin-UNETR (MONAI implementations, optional pretrained weights).
- **Teacher-student distillation**: Dice + CE segmentation loss combined with temperature-scaled KL distillation from a full-modality teacher.
- 
## 📌 Pipeline Overview
![Pipeline](img/2_overall_framework.png)

## 📌 Dual Prompt Injection mechanism
![Dual Prompt](img/3_Prompt_injection.png)

## Repository Structure

```
.
├── train_teacher.py           # Stage 1: train the full-modality teacher
├── train_prompt_distill.py    # Stage 2: prompt distillation (unified / per_subset)
├── trainer_prompt.py          # Distillation trainer implementation
├── evaluate_prompt.py         # Evaluation for per_subset (separate) checkpoints
├── evaluate_unified.py        # Multi-subset evaluation for unified checkpoints
├── prompt_modules.py          # General/expert prompt pools and injection wrappers
├── networks_monai.py          # UNETR / Swin-UNETR wrappers (MONAI)
├── vnet_original.py           # VNet backbone
├── datasets.py                # BraTS dataset and augmentations
├── loss.py                    # Dice+CE and KD losses
├── evaluate.py                # Sliding-window inference and metrics (Dice/HD95/Sen/Spe)
├── download_pretrained_weights.py
├── train.sh                   # Convenience wrapper for the commands below
├── requirements.txt
└── utils.py, cuda_memory_utils.py
```

## Installation

```bash
conda create -n prompt-kd python=3.9 -y
conda activate prompt-kd
pip install -r requirements.txt

# Optional: pre-download MONAI pretrained weights for UNETR / Swin-UNETR
python download_pretrained_weights.py --model all --cache-dir ./pretrained_models
```

Tested with PyTorch >= 1.12 and MONAI >= 1.3 on a single NVIDIA GPU (>= 24 GB recommended for Swin-UNETR).

## Data Preparation

We use the BraTS dataset. Each case must be preprocessed into a single NumPy array of shape `[5, D, H, W]` saved as `<case_id>.npy`:

- Channels 0-3: T1, T2, T1ce, FLAIR (co-registered, skull-stripped, z-score normalized per volume);
- Channel 4: segmentation label with values `{0: background, 1: NCR/NET, 2: edema, 3: enhancing tumor}` (the original BraTS label 4 is remapped to 3);
- Spatial size must be at least the training crop `128 x 128 x 128`.

Organize the data directory as:

```
data/
├── brats2020/
│   ├── <case_id>.npy
│   └── ...
├── train_list.txt        # one case_id per line
├── val_list.txt
└── test_list.txt
├── brats2018/brats2015
```

Modality indices used throughout the code: `T1=0, T2=1, T1ce=2, FLAIR=3`. Subsets can be specified by name (`"T1+T2"`, case-insensitive) or by index (`"0,1"`).

## Stage 1: Train the Full-Modality Teacher

The teacher always takes all 4 modalities as input.

```bash
# Swin-UNETR teacher (with MONAI pretrained weights)
python train_teacher.py \
    --model_type swin_unetr \
    --pretrained \
    --batch_size 2 \
    --max_epoch 500 \
    --lr 0.0005 \
    --data_dir ./data \
    --log_dir ./log/teacher

# UNETR teacher
python train_teacher.py \
    --model_type unetr \
    --pretrained \
    --batch_size 2 \
    --max_epoch 500 \
    --lr 0.0005 \
    --data_dir ./data \
    --log_dir ./log/teacher

# VNet teacher (trained from scratch)
python train_teacher.py \
    --model_type vnet \
    --batch_size 4 \
    --max_epoch 1000 \
    --lr 0.001 \
    --data_dir ./data \
    --log_dir ./log/teacher
```

The best checkpoint is saved to `./log/teacher/<model_type>_teacher/model/best_model.pth`. Use `--gradient_accumulation_steps` to enlarge the effective batch size when GPU memory is limited.

## Stage 2: Prompt Distillation

`train_prompt_distill.py` supports both training paradigms via `--training_mode`.

### Option A: Unified Training (one model for all 15 subsets)

Each iteration samples a random non-empty modality subset, zero-fills the missing channels (the student always has 4 input channels), and jointly updates the general prompts plus all 15 expert prompt pools.

```bash
python train_prompt_distill.py \
    --training_mode unified \
    --model_type unetr \
    --unified_sample_strategy uniform_subset \
    --unified_val_subsets default \
    --student_modalities T1 \
    --teacher_ckpt_path ./log/teacher/unetr_teacher/model/best_model.pth \
    --freeze_stu_encoder \
    --general_prompt_length 5 \
    --expert_prompt_length 5 \
    --batch_size 4 \
    --lr 0.001 \
    --max_epoch 1000 \
    --data_dir ./data \
    --log_dir ./log/prompt_distill
```

Unified-mode options:

- `--unified_sample_strategy`: subset sampling curriculum.
  - `uniform_subset` (default): uniform over all 15 non-empty subsets;
  - `uniform_size`: first sample the subset size (1-4) uniformly, then a subset of that size;
  - `fixed_one_missing`: always drop exactly one modality (easier warm-up).
- `--unified_val_subsets`: which subsets to validate each validation epoch.
  - `default`: only the subset given by `--student_modalities` (fast per-epoch signal);
  - `all`: all 15 subsets (thorough but slow);
  - or an explicit list, e.g. `"T1,T1+T2,FLAIR"`.

In unified mode, `--student_modalities` does not restrict training; it only sets the default validation subset.

The run directory is created at `./log/prompt_distill/<model_type>_unified_<strategy>/`, with the best checkpoint at `.../model/best_model.pth`.

### Option B: Separate Training (one specialist per subset)

Each run targets a fixed modality subset. The student's input channels equal the subset size, and only the expert prompt pool of that subset is trained (all others are frozen).

```bash
# Single modality (T1)
python train_prompt_distill.py \
    --training_mode per_subset \
    --model_type unetr \
    --student_modalities T1 \
    --teacher_ckpt_path ./log/teacher/unetr_teacher/model/best_model.pth \
    --freeze_stu_encoder \
    --batch_size 4 \
    --lr 0.001 \
    --max_epoch 1000 \
    --data_dir ./data \
    --log_dir ./log/prompt_distill

# Two modalities (T1+T2)
python train_prompt_distill.py \
    --training_mode per_subset \
    --model_type unetr \
    --student_modalities "T1+T2" \
    --teacher_ckpt_path ./log/teacher/unetr_teacher/model/best_model.pth \
    --freeze_stu_encoder \
    --batch_size 4 \
    --lr 0.001 \
    --max_epoch 1000 \
    --data_dir ./data \
    --log_dir ./log/prompt_distill

# Three modalities (T1+T2+FLAIR) with Swin-UNETR
python train_prompt_distill.py \
    --training_mode per_subset \
    --model_type swin_unetr \
    --student_modalities "T1+T2+FLAIR" \
    --teacher_ckpt_path ./log/teacher/swin_unetr_teacher/model/best_model.pth \
    --freeze_stu_encoder \
    --general_prompt_length 10 \
    --expert_prompt_length 10 \
    --batch_size 2 \
    --lr 0.0005 \
    --max_epoch 800 \
    --data_dir ./data \
    --log_dir ./log/prompt_distill
```

The run directory is created at `./log/prompt_distill/<model_type>_<subset_key>/` (e.g. `unetr_T1+T2/`), with the best checkpoint at `.../model/best_model.pth`. To cover all 15 subsets in this paradigm, launch 15 runs (see `train.sh`).

### Notes on cross-architecture distillation

The teacher architecture is inferred from the checkpoint metadata; use `--teacher_model_type {vnet,unetr,swin_unetr}` to override it explicitly when distilling across architectures (e.g. a Swin-UNETR teacher into a UNETR student).

## Evaluation

### Unified checkpoints

`evaluate_unified.py` rebuilds the 4-channel student, iterates over the requested subsets with per-subset zero-filling, and reports Dice/HD95/Sensitivity/Specificity for WT/TC/ET. It writes a CSV summary and a Markdown results table to `--output_path`.

```bash
python evaluate_unified.py \
    --model_path ./log/prompt_distill/unetr_unified_uniform_subset/model/best_model.pth \
    --data_dir ./data \
    --subsets all \
    --output_path ./results/unified_eval
```

`--subsets` accepts `all` (15 subsets), `size1` ... `size4`, or an explicit list such as `"T1,T1+T2,FLAIR+T1ce"`.

### Separate (per_subset) checkpoints

`evaluate_prompt.py` reads the modality subset from the checkpoint and evaluates on the test list. Add `--save_vis` to export predictions as NIfTI volumes.

```bash
python evaluate_prompt.py \
    --model_path ./log/prompt_distill/unetr_T1/model/best_model.pth \
    --data_dir ./data \
    --output_path ./results/per_subset_T1
```

## Command-Line Reference (train_prompt_distill.py)

| Argument | Default | Description |
|---|---|---|
| `--training_mode` | `per_subset` | `unified` or `per_subset` (see above) |
| `--model_type` | `swin_unetr` | `vnet`, `unetr`, or `swin_unetr` |
| `--student_modalities` | `T1` | Fixed subset in `per_subset` mode; default validation subset in `unified` mode |
| `--teacher_ckpt_path` | `''` | Path to the Stage-1 teacher checkpoint |
| `--teacher_model_type` | inferred | Override teacher architecture |
| `--freeze_stu_encoder` / `--train_stu_encoder` | freeze | Freeze or fine-tune the student encoder |
| `--general_prompt_length` | `5` | Length of the shared general prompts |
| `--expert_prompt_length` | `5` | Length of each subset-specific expert prompt pool |
| `--seg_weight` | `1.0` | Weight of the Dice+CE segmentation loss |
| `--kd_weight` | `10.0` | Weight of the KL distillation loss |
| `--temperature` | `10.0` | Distillation temperature (standard T^2 scaling) |
| `--batch_size` | `4` | Batch size |
| `--lr` | `0.001` | Learning rate (Adam) |
| `--max_epoch` | `1000` | Number of training epochs |
| `--unified_sample_strategy` | `uniform_subset` | Subset sampling strategy (unified mode) |
| `--unified_val_subsets` | `default` | Validation subsets (unified mode) |
| `--seed` | `42` | Random seed |
| `--gpu` | `0` | GPU id |
| `--resume` / `--ckpt_path` | off | Resume training from a checkpoint |

## Practical Tips

- **Reproducibility**: all training scripts fix the random seed (default 42) via `--seed`.
- **GPU memory**: Swin-UNETR needs noticeably more memory than UNETR; reduce `--batch_size` (and learning rate accordingly) or fall back to UNETR if you hit OOM.
- **Validation schedule**: the distillation trainer starts validating after the first quarter of training and validates every 5 epochs, so `best_model.pth` appears only after that point.
- **Pretrained weights**: UNETR/Swin-UNETR encoders are initialized from MONAI weights cached in `--cache_dir` (default `./pretrained_models`); if the automatic download fails, run `download_pretrained_weights.py` or fetch them manually from the MONAI Model Zoo.
- **Freezing the encoder** (`--freeze_stu_encoder`, the default) trains only the prompts and decoder, which is both faster and more stable than full fine-tuning in our experiments.

## License

This project is released under the Apache License 2.0 (see `LICENSE`).

## Acknowledgments

- [MONAI](https://monai.io/) for the UNETR / Swin-UNETR implementations and pretrained weights.
- The [BraTS challenge](https://www.med.upenn.edu/cbica/brats/) organizers for the dataset.

## Citation

If you find this repository useful, please cite our paper (citation to be added upon publication).
