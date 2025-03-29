# Antibody BPE Tokenizer

A minimal Byte-Pair Encoding (BPE) tokenizer specifically designed for antibody heavy and light chain sequences. This tokenizer implements vocabulary learning from scratch and handles special tokens for antibody sequence processing.

## Features

- Custom BPE tokenization for antibody sequences
- Handles heavy and light chain sequences with chain break token
- Special tokens support (`<cls>`, `<eos>`, `<pad>`, `<mask>`, `<unk>`)
- Vocabulary learning from scratch
- Save and load tokenizer configurations
- Regex-based pre-tokenization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/deepbiolab/antibody-bpe-tokenizer.git
cd antibody-bpe-tokenizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training a New Tokenizer

```bash
python main.py --train \
    --data_file datasets/processed/sabdab_sequences.csv \
    --vocab_size 1000
```

### Loading a Pre-trained Tokenizer

```bash
python main.py \
    --data_file datasets/processed/sabdab_sequences.csv \
    --vocab_size 1000
```

### Command Line Arguments

- `--train`: Enable training mode (optional)
- `--data_file`: Path to the input data file (default: datasets/processed/sabdab_sequences.csv)
- `--vocab_size`: Size of the vocabulary (default: 1000)
- `--checkpoint_name`: Name of the checkpoint file (default: antibody_tokenizer)

## Input Data Format

The input CSV file should contain the following columns:
- `heavy_chain`: Heavy chain amino acid sequence
- `light_chain`: Light chain amino acid sequence

The tokenizer will automatically combine these sequences using a chain break token (`|`).

## Code Examples

### Basic Usage

```python
from src import AntibodyTokenizer

# Initialize tokenizer
tokenizer = AntibodyTokenizer(vocab_size=1000)

# Train tokenizer
tokenizer.train(training_text, verbose=True)

# Encode sequence
sequence = "QVQLVQSGAE|DIQMTQSPSS"
encoded = tokenizer.encode(sequence)
decoded = tokenizer.decode(encoded)

print(f"Original: {sequence}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
```

### Save and Load Tokenizer

```python
# Save tokenizer
tokenizer.save_tokenizer(prefix="my_tokenizer")

# Load tokenizer
loaded_tokenizer = AntibodyTokenizer(vocab_size=1000)
loaded_tokenizer.load_tokenizer(prefix="my_tokenizer")
```

## Project Structure

```
antibody-bpe-tokenizer/
├── src/
│   ├── __init__.py
│   ├── basic_tokenizer.py    # Base BPE implementation
│   ├── regex_tokenizer.py    # Regex-based pre-tokenization
│   └── antibody_tokenizer.py # Main tokenizer class
├── main.py                   # Training and testing script
└── requirements.txt          # Project dependencies
```

## Implementation Details

The tokenizer consists of three main components:

1. **BasicTokenizer**: Implements core BPE algorithm
   - Byte-level tokenization
   - Merge rules learning
   - Vocabulary building

2. **RegexTokenizer**: Extends BasicTokenizer
   - Adds regex-based pre-tokenization
   - Handles special characters and patterns

3. **AntibodyTokenizer**: Main tokenizer class
   - Special tokens management
   - Chain break token handling
   - Save/load functionality
   - Encoding/decoding with special tokens

## Future Work

### Motif-Aware Tokenization

The current tokenizer implementation focuses on statistical patterns in the sequence data. However, antibody sequences contain biologically significant amino acid patterns called motifs, which carry important semantic information. Future improvements could include:

1. **Motif Analysis**
   - Add tools for identifying common motifs in the learned vocabulary
   - Analyze correlation between learned merge pairs and known antibody motifs
   - Visualize motif coverage in the tokenization results

2. **Extended Features**
   - Support for CDR (Complementarity Determining Regions) identification
   - Framework region-aware tokenization


## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments
- Inspired by andrej Karpathy's BPE implementation: https://github.com/karpathy/minbpe
- Antibody Database (SAbDab) for the antibody sequence dataset.
