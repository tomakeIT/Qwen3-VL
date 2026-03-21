#!/bin/bash
set -euo pipefail

# Source should be one specific checkpoint-* directory.
SRC_CKPT_DIR=${SRC_CKPT_DIR:-/home/jialeng/Qwen3-VL/qwen-vl-finetune/output/lerobot_Robocasa_X7s_6tasks_RTX6000/checkpoint-1000}

# Destination root for universal-resume run. Keep checkpoint-* naming for HF auto-resume.
DST_ROOT_DIR=${DST_ROOT_DIR:-/home/jialeng/Qwen3-VL/qwen-vl-finetune/output/lerobot_Robocasa_X7s_6tasks_RTX6000_resume}
DST_CKPT_DIR="${DST_ROOT_DIR}/$(basename "${SRC_CKPT_DIR}")"

if [[ ! -d "${SRC_CKPT_DIR}" ]]; then
  echo "Source checkpoint directory not found: ${SRC_CKPT_DIR}"
  exit 1
fi

if [[ ! -f "${SRC_CKPT_DIR}/latest" ]]; then
  echo "Missing latest file under: ${SRC_CKPT_DIR}"
  exit 1
fi

STEP_TAG=$(<"${SRC_CKPT_DIR}/latest")
STEP_TAG=${STEP_TAG//$'\r'/}
STEP_TAG=${STEP_TAG//$'\n'/}

mkdir -p "${DST_CKPT_DIR}"
echo "Using step tag: ${STEP_TAG}"
echo "Converting to: ${DST_CKPT_DIR}/${STEP_TAG}"

conda activate qwen
python -m deepspeed.checkpoint.ds_to_universal \
  --input_folder "${SRC_CKPT_DIR}/${STEP_TAG}" \
  --output_folder "${DST_CKPT_DIR}/${STEP_TAG}" \
  --inject_missing_state

# HF Trainer resume requires these metadata files under checkpoint-* root.
for f in trainer_state.json scheduler.pt training_args.bin latest; do
  if [[ -f "${SRC_CKPT_DIR}/${f}" ]]; then
    cp -f "${SRC_CKPT_DIR}/${f}" "${DST_CKPT_DIR}/${f}"
  fi
done

shopt -s nullglob
for f in "${SRC_CKPT_DIR}"/rng_state_*.pth; do
  cp -f "${f}" "${DST_CKPT_DIR}/"
done
shopt -u nullglob

echo "Done. Universal checkpoint ready at: ${DST_CKPT_DIR}"