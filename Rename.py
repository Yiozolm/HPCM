import torch
import os
import re
import math
import argparse

patterns = {
    r'^g_a\.branch\.': 'g_a.',
    r'^g_s\.branch\.': 'g_s.',
    r'^h_a\.branch\.': 'h_a.',
    r'^h_s\.branch\.': 'h_s.',
    r'^entropy_estimation\.': 'entropy_bottleneck.',
    r'^scale_table$': 'gaussian_conditional.scale_table',
    r'^quantized_cdf_z$': 'entropy_bottleneck._quantized_cdf',
    r'^cdf_length_z$': 'entropy_bottleneck._cdf_length',
    r'^offset_z$': 'entropy_bottleneck._offset',
    r'^quantized_cdf_y$': 'gaussian_conditional._quantized_cdf',
    r'^cdf_length_y$': 'gaussian_conditional._cdf_length',
    r'^offset_y$': 'gaussian_conditional._offset',
}

SCALES_MIN = 0.12  # 0.11 in compressai, the author use 0.12
SCALES_MAX = 256
SCALES_LEVELS = 60 # 64 in compressai, the author use 60
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def rename(origin_path, output_path) -> None:
    state_dict = torch.load(origin_path, map_location=DEVICE, weights_only=True)
    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key
        for pattern, replacement in patterns.items():
            if re.match(pattern, new_key):
                new_key = re.sub(pattern, replacement, new_key)
                break
        
        if key == 'entropy_estimation.beta':
            new_key = 'entropy_bottleneck.beta'
        elif key == 'entropy_estimation.scale_lower_bound.bound':
            new_key = 'entropy_bottleneck.lower_bound_scale.bound'
        elif key == 'entropy_estimation.likelihood_lower_bound.bound':
            new_key = 'entropy_bottleneck.likelihood_lower_bound.bound'
        
        new_state_dict[new_key] = value

    new_state_dict['entropy_bottleneck.scale_table'] = new_state_dict['scales_hyper'].view(-1)
    new_state_dict['entropy_bottleneck.scale_bound'] = torch.Tensor([0.12]) 
    new_state_dict['gaussian_conditional.beta'] = new_state_dict['entropy_bottleneck.beta']
    new_state_dict['gaussian_conditional.scale_bound'] = torch.Tensor([0.12]) 
    new_state_dict['gaussian_conditional.likelihood_lower_bound.bound'] = new_state_dict['entropy_bottleneck.likelihood_lower_bound.bound']
    new_state_dict['gaussian_conditional.lower_bound_scale.bound'] = new_state_dict['entropy_bottleneck.lower_bound_scale.bound'] 

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(new_state_dict, output_path)


def main():
    parser = argparse.ArgumentParser(description='Rename checkpoint files')
    parser.add_argument('--origin_path', '-i', type=str, required=True,
                       help='Path to original checkpoints directory')
    parser.add_argument('--output_path', '-o', type=str, required=True,
                       help='Path to output renamed checkpoints directory')
    
    args = parser.parse_args()
    
    origin_path = args.origin_path
    output_path = args.output_path
    
    files = os.listdir(origin_path)
    for file in files:
        try:
            rename(os.path.join(origin_path, file), os.path.join(output_path, file))
            print(f'{file} renamed successfully')
        except Exception as e:
            print(e)

if __name__ == '__main__':
    main()
