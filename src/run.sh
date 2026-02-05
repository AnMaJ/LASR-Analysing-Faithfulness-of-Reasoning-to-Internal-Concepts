#!/bin/bash

#Chaning the paths to cache in bitbucket
export HF_HOME="/vol/bitbucket/${USER}/.cache"
export HF_DATASETS_CACHE="/vol/bitbucket/${USER}/.cache/huggingface/datasets"
export DIFFUSERS_CACHE="/vol/bitbucket/${USER}/.cache/huggingface/diffusers"
export MODELSCOPE_CACHE="/vol/bitbucket/m24/.cache/modelscope"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

echo "Environment variables set for Hugging Face cache and CUDA devices."

DATASET_NAME="ECQA"
DATASET_PATH="/vol/bitbucket/m24/Concept_Neuron_Localisation_my_idea/Investigating_concept_neurons/LASR-Analysing-Faithfulness-of-Reasoning-to-Internal-Concepts/datasets/ecqa/data/validation-00000-of-00001.parquet"
OUTPUT_PATH="/vol/bitbucket/m24/Concept_Neuron_Localisation_my_idea/Investigating_concept_neurons/LASR-Analysing-Faithfulness-of-Reasoning-to-Internal-Concepts/outputs"
MODEL_NAME_OR_PATH="google/gemma-3-4b-it"
SAE_STORE_PATH="google/gemma-scope-2-4b-pt"
TARGET_LAYER=22
SAE_WIDTH="256k"
L0_SPARSITY="medium"
LOGS_PATH="/vol/bitbucket/m24/Concept_Neuron_Localisation_my_idea/Investigating_concept_neurons/LASR-Analysing-Faithfulness-of-Reasoning-to-Internal-Concepts/logs/analysis_gemma_3_4b_it_target_layer_${TARGET_LAYER}_sae_width_${SAE_WIDTH}_l0_sparsity_${L0_SPARSITY}.log"

mkdir -p "$(dirname "$LOGS_PATH")"


echo "==============================="
echo "Starting feature extraction analysis..."
echo "Dataset Name: $DATASET_NAME"
echo "Dataset Path: $DATASET_PATH"
echo "Output Path: $OUTPUT_PATH"
echo "Model Name or Path: $MODEL_NAME_OR_PATH"
echo "SAE Store Path: $SAE_STORE_PATH"
echo "Target Layer: $TARGET_LAYER"
echo "SAE Width: $SAE_WIDTH"
echo "L0 Sparsity: $L0_SPARSITY"
echo "==============================="

python3 -m cli \
    --dataset-name "$DATASET_NAME" \
    --dataset-path "$DATASET_PATH" \
    --output-path "$OUTPUT_PATH" \
    --model-name-or-path "$MODEL_NAME_OR_PATH" \
    --sae-store-path "$SAE_STORE_PATH" \
    --target-layer "$TARGET_LAYER" \
    --sae-width "$SAE_WIDTH" \
    --l0-sparsity "$L0_SPARSITY" \
    > "$LOGS_PATH" 2>&1

echo "Analysis completed. Logs saved to $LOGS_PATH"
