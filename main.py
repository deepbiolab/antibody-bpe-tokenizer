import os
import pandas as pd
import argparse

from src import AntibodyTokenizer

def load_and_preprocess_data(file_path: str) -> tuple:
    """
    Load and preprocess the dataset
    Returns: (complete training text, test sequence tuple)
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Preprocessing: combine heavy chain and light chain
    df['full_sequence'] = df['heavy_chain'] + '|' + df['light_chain']
    
    # Create training text (all sequences joined by $)
    training_text = '$'.join(df['full_sequence'].tolist())
    
    # Create test data
    # 1. Randomly select a complete antibody sequence
    test_full = df['full_sequence'].iloc[0]
    # 2. Corresponding heavy chain
    test_heavy = df['heavy_chain'].iloc[0]
    # 3. Corresponding light chain
    test_light = df['light_chain'].iloc[0]
    
    return training_text, (test_full, test_heavy, test_light)

def test_tokenizer(tokenizer: AntibodyTokenizer, test_sequences: tuple, label: str = ""):
    """Test tokenizer's encoding and decoding functionality"""
    test_full, test_heavy, test_light = test_sequences
    
    print(f"\n=== Testing Tokenizer {label} ===")
    
    # Test complete antibody sequence
    print("\n1. Testing full antibody sequence:")
    print(f"Original: {test_full}")
    encoded_full = tokenizer.encode(test_full)
    decoded_full = tokenizer.decode(encoded_full)
    print(f"Encoded: {encoded_full}")
    print(f"Decoded: {decoded_full}")
    print(f"Token count: {len(encoded_full)}")
    
    # Test heavy chain sequence
    print("\n2. Testing heavy chain sequence:")
    print(f"Original: {test_heavy}")
    encoded_heavy = tokenizer.encode(test_heavy)
    decoded_heavy = tokenizer.decode(encoded_heavy)
    print(f"Encoded: {encoded_heavy}")
    print(f"Decoded: {decoded_heavy}")
    print(f"Token count: {len(encoded_heavy)}")
    
    # Test light chain sequence
    print("\n3. Testing light chain sequence:")
    print(f"Original: {test_light}")
    encoded_light = tokenizer.encode(test_light)
    decoded_light = tokenizer.decode(encoded_light)
    print(f"Encoded: {encoded_light}")
    print(f"Decoded: {decoded_light}")
    print(f"Token count: {len(encoded_light)}")
    
    # Test special tokens
    print("\n4. Special tokens information:")
    print(f"CLS token: {tokenizer.cls_token} (id: {tokenizer.cls_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print(f"Chain break token: {tokenizer.chain_break_token} (id: {tokenizer.chain_break_token_id})")
    print(f"Total vocabulary size: {tokenizer.vocab_size}")

    # Test simple chain break encoding
    chain_break_test = "A|G"
    encoded_chain_break = tokenizer.encode(chain_break_test)
    decoded_chain_break = tokenizer.decode(encoded_chain_break)
    print(f"\n5. Chain break test:")
    print(f"Original: {chain_break_test}")
    print(f"Encoded: {encoded_chain_break}")
    print(f"Decoded: {decoded_chain_break}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and test Antibody Tokenizer')
    parser.add_argument('--data_file', type=str, 
                       default='datasets/processed/sabdab_sequences.csv',
                       help='Path to the input data file (default: datasets/processed/sabdab_sequences.csv)')
    parser.add_argument('--vocab_size', type=int, 
                       default=1000,
                       help='Size of the vocabulary (default: 1000)')
    parser.add_argument('--train', action='store_true',
                       help='Train the tokenizer. If not specified, load from existing checkpoint')
    parser.add_argument('--checkpoint_name', type=str,
                       default='antibody_tokenizer',
                       help='Name of the checkpoint file (without .json extension)')
    
    args = parser.parse_args()
    
    print("=== Antibody Tokenizer Demo ===")
    print(f"Mode: {'Training' if args.train else 'Loading'}")
    
    # Initialize tokenizer
    tokenizer = AntibodyTokenizer(vocab_size=args.vocab_size)
    
    if args.train:
        # Training mode
        print("\nLoading and preprocessing data...")
        training_text, test_sequences = load_and_preprocess_data(args.data_file)
        print(f"Total training text length: {len(training_text)}")
        
        print("\nTraining tokenizer...")
        tokenizer.train(training_text, verbose=True)
        
        print("\nSaving tokenizer...")
        tokenizer.save_tokenizer(prefix=args.checkpoint_name)
        
    else:
        # Loading mode
        checkpoint_path = os.path.join("checkpoints", f"{args.checkpoint_name}.json")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found at {checkpoint_path}. "
                "Please train the tokenizer first using --train flag"
            )
        
        print("\nLoading tokenizer from checkpoint...")
        tokenizer.load_tokenizer(prefix=args.checkpoint_name)
        
        # Load test data for evaluation
        print("\nLoading test data...")
        _, test_sequences = load_and_preprocess_data(args.data_file)
    
    # Test tokenizer
    test_tokenizer(tokenizer, test_sequences)
    
    # Additional test with specific sequence
    print("\n=== Testing with specific sequence ===")
    test_text = "QVQLVQSGAE|DIQMTQSPSS"
    print(f"\nTest sequence: {test_text}")
    
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Sequence preserved: {test_text == decoded}")

if __name__ == "__main__":
    main()