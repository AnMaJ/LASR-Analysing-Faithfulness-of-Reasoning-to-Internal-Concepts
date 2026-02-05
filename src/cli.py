import argparse
from loading_model import load_model_and_tokenizer
from loading_dataset import ReasoningDataset
from sae import get_top_k_features_from_sae
from utils import save_run_config

def main():
    parser = argparse.ArgumentParser(description="Analyze faithfulness of reasoning to internal concepts.")
    
    parser.add_argument(
        '--dataset-name',
        type=str,
        required=True,
        help='Name of the dataset to use.',
        choices=['ECQA', 'e-SNLI', 'BBQ']
    )
    
    parser.add_argument(
        '--dataset-path', 
        type=str, 
        required=True, 
        help='Path to the input dataset')
    
    parser.add_argument(
        '--output-path', 
        type=str, 
        required=True, 
        help='Path to the directory to store the results.')
    
    parser.add_argument(
        '--model-name-or-path',
        type=str,
        required=True,
        help='Name or path of the pre-trained model to use.'
    )
    
    parser.add_argument(
        '--sae-store-path',
        type=str,
        required=True,
        help='Path to the folder containing the pre-trained SAEs.'
    )
    
    parser.add_argument(
        '--target-layer',
        type=int,
        required=True,
        help='Target layer number for analysis.'
    )
    
    parser.add_argument(
        '--sae-width',
        type=str,
        required=True,
        help='Width of the SAE to use.',
        choices=['16k', '64k', '256k', '512k', '1m']
    )
    
    parser.add_argument(
        '--l0-sparsity',
        type=str,
        default="medium",
        help='L0 sparsity parameter used during SAE training.'
    )
    
    args = parser.parse_args()
    save_run_config(args, f"{args.output_path}/run_config.txt")
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)    
    # implementing for only a sample of the dataset for now
    dataset = ReasoningDataset(name=args.dataset_name, path=args.dataset_path)
    sample_input = dataset[0]["q_text"] # just checking for questions for now
    top_activations, top_latents, sae_activations = get_top_k_features_from_sae(args, sample_input, model, tokenizer, k=10)
    
    return 


if __name__ == "__main__":
    main()
    
    