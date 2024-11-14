import torch
import argparse
from pathlib import Path
from collections import OrderedDict
import json

def inspect_state_dict(checkpoint_path):
    """
    Load and inspect a PyTorch checkpoint, displaying detailed information about its structure
    """
    print(f"\nInspecting checkpoint: {checkpoint_path}")
    print("-" * 80)
    
    # Load the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✓ Successfully loaded checkpoint\n")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {str(e)}")
        return
    
    # Determine if it's a complete checkpoint or just state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("Found complete checkpoint with additional information:")
        for key in checkpoint.keys():
            if key != 'model_state_dict':
                print(f"- {key}")
        print("\nAnalyzing model_state_dict:")
    else:
        state_dict = checkpoint
        print("Found state_dict only checkpoint")

    # Analyze the state dict
    print("\nLayer Structure:")
    print("-" * 80)
    
    # Group parameters by their prefix
    parameter_groups = OrderedDict()
    for key, tensor in state_dict.items():
        # Get the main component name (everything before the first dot)
        main_component = key.split('.')[0]
        if main_component not in parameter_groups:
            parameter_groups[main_component] = []
        parameter_groups[main_component].append((key, tensor))

    # Print detailed information for each group
    total_parameters = 0
    for component, parameters in parameter_groups.items():
        print(f"\n{component}:")
        print("-" * 40)
        
        for key, tensor in parameters:
            shape = list(tensor.shape)
            num_params = torch.tensor(shape).prod().item()
            total_parameters += num_params
            print(f"Layer: {key}")
            print(f"  Shape: {shape}")
            print(f"  Parameters: {num_params:,}")
            print(f"  Dtype: {tensor.dtype}")
            print()

    print("-" * 80)
    print(f"Total Parameters: {total_parameters:,}")
    
    # Save the analysis to a JSON file
    output_path = Path(checkpoint_path).with_suffix('.analysis.json')
    analysis = {
        "checkpoint_path": str(checkpoint_path),
        "total_parameters": total_parameters,
        "layers": {
            component: [
                {
                    "name": key,
                    "shape": list(tensor.shape),
                    "parameters": torch.tensor(list(tensor.shape)).prod().item(),
                    "dtype": str(tensor.dtype)
                }
                for key, tensor in parameters
            ]
            for component, parameters in parameter_groups.items()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nDetailed analysis saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Inspect PyTorch model checkpoints')
    parser.add_argument('checkpoint_path', type=str, help='Path to the checkpoint file')
    args = parser.parse_args()
    
    inspect_state_dict(args.checkpoint_path)

if __name__ == '__main__':
    main()