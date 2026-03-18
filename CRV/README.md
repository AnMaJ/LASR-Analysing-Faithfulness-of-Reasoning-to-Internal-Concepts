# Verifying Chain-of-Thought Reasoning via its Computational Graph

This is the official code repository for the paper **"Verifying Chain-of-Thought Reasoning via its Computational Graph"**.


[![arXiv](https://img.shields.io/badge/arXiv-2510.09312-b31b1b.svg)](https://arxiv.org/pdf/2510.09312) [![Model](https://img.shields.io/badge/HuggingFace-Model-blue)](https://huggingface.co/facebook/crv-8b-instruct-transcoders)
 [![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-FFD21E)](https://huggingface.co/datasets/facebook/crv)




## Installation
To create the environment and install dependencies:
```bash
conda env create -f environment.yml
conda activate crv
```

## Data
All datasets used in the paper are located in the `data/` folder. You can also download the datasets directly from [Hugging Face](https://huggingface.co/datasets/facebook/crv).

## Baselines
We compare CRV against black-box (MaxProb, PPL, Entropy, Temperature Scaling, Energy) and gray-box (Chain-of-Embedding, CoT-Kinetics) methods.

To reproduce these baselines, please use our fork of the [Chain-of-Embedding repository](https://github.com/Alsace08/Chain-of-Embedding):

1. Clone the repository:
```bash
git clone https://github.com/ncancedda/Chain-of-Embedding/
cd Chain-of-Embedding/
git checkout fair-mir
```
2. Install the required dependencies:
```bash
conda create -n coeeval python=3.10
conda activate coeeval
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

3. Run the baselines:

```bash
python main.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset /path/to/dataset \
    --print_model_parameter \
    --save_output \
    --save_hidden_states \
    --save_coe_score \
    --save_cotk_score \
    --use_cached_outputs \
    --token_aggregation "average" \
    --cotk_gamma 0.8 \
    --no_dialogs
```
4. Evaluate results:
Results are saved to `OutputInfo/`. Run the evaluation script to compute AUROC, AUPR, and FPR@95:
```bash
python ./Evaluation/eval.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset /path/to/dataset \
    --pos_label 0 \
    --token_aggregation "average"
```
*Note: We set `pos_label=0` because we treat the incorrect label as the positive class for our metrics.*

## Models and Transcoders

This work utilizes **Llama 3.1 8B Instruct** with MLPs replaced by trained **Top-K Transcoders**.

- **Weights**: The per-layer transcoder weights can be downloaded from [Hugging Face](https://huggingface.co/facebook/crv-8b-instruct-transcoders).
- **Usage**: To load and use the transcoders, please refer to the `models/` directory for instruction.

## Attribution Graphs
We utilize the [Circuit Tracer](https://github.com/safety-research/circuit-tracer) library to construct the attribution graphs. Detailed instructions and the specific scripts used to generate graphs for our datasets are provided in the `attribution/` folder.

## Feature Extraction
Once attribution graphs are generated, use the extraction script to convert graph structures into tabular features for classification.

- **Script**: `features/feature_extraction.py`
- **Documentation**: See `features/README.md` for details.

## Verification
Finally, with all features extracted, you can train the diagnostic classifier. We recommend to use the `scikit-learn` library for this purpose, as they provide implementation for a wide selection of classification methods.

## Citation
If you find this work useful, please consider citing our paper:
```
@article{zhao2025verifying,
      title={Verifying Chain-of-Thought Reasoning via Its Computational Graph},
      author={Zheng Zhao and Yeskendir Koishekenov and Xianjun Yang and Naila Murray and Nicola Cancedda},
      year={2025},
      eprint={2510.09312},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.09312},
}
```

## License
This project is licensed under the CC-BY-NC-4.0 License.
