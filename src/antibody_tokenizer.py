import os
import json
from typing import List
from .regex_tokenizer import RegexTokenizer

class AntibodyTokenizer:
    def __init__(
        self,
        unk_token="<unk>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        chain_break_token="|",
        vocab_size=266,
        checkpoint_dir="checkpoints"
    ):
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Initialize special tokens dictionary
        self._special_tokens = {
            'unk_token': unk_token,
            'cls_token': cls_token,
            'pad_token': pad_token,
            'mask_token': mask_token,
            'eos_token': eos_token,
        }
        self._cb_token = chain_break_token
        
        # Count of special tokens
        self._num_special_tokens = len(self._special_tokens)
        
        # Adjust vocab_size to account for special tokens
        self._total_vocab_size = vocab_size
        self._base_vocab_size = vocab_size - self._num_special_tokens
        
        self._base_tokenizer = RegexTokenizer()
        
        # Initialize special token IDs (starting after byte tokens)
        self._special_token_ids = {
            token: self._base_vocab_size + i 
            for i, token in enumerate(self._special_tokens.values())
        }

    def train(self, text: str, verbose: bool = False):
        """Train the base tokenizer on input text"""
        self._base_tokenizer.train(text, self._base_vocab_size, verbose)
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text using the base tokenizer"""
        base_ids = self._base_tokenizer.encode(text)
        if add_special_tokens:
            return [self._special_token_ids[self._special_tokens['cls_token']]] + \
                   base_ids + \
                   [self._special_token_ids[self._special_tokens['eos_token']]]
        return base_ids
        
    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token ids back to text"""
        if not skip_special_tokens:
            # Convert special tokens to their string representation
            tokens = []
            for id in ids:
                if id >= self._base_vocab_size:
                    # This is a special token
                    for token, token_id in self._special_token_ids.items():
                        if token_id == id:
                            tokens.append(token)
                            break
                else:
                    # This is a regular token
                    tokens.append(self._base_tokenizer.decode([id]))
            return ''.join(tokens)
        
        # Skip special tokens
        regular_ids = [id for id in ids if id < self._base_vocab_size]
        return self._base_tokenizer.decode(regular_ids)
    
    @property
    def cls_token(self):
        return self._get_token("cls_token")

    @property
    def cls_token_id(self):
        return self._get_token_id(self.cls_token)
    
    @property
    def eos_token(self):
        return self._get_token("eos_token")

    @property
    def eos_token_id(self):
        return self._get_token_id(self.eos_token)
    
    @property
    def chain_break_token(self):
        return self._cb_token

    @property
    def chain_break_token_id(self):
        return ord(self.chain_break_token)
    
    @property
    def vocab_size(self):
        return self._total_vocab_size
    
    def _get_token(self, token_name: str) -> str:
        token_str = self._special_tokens[token_name]
        return token_str
    
    def _get_token_id(self, token) -> int:
        token_id = self._special_token_ids[token]
        return token_id

    def _bytes_to_readable(self, byte_seq: bytes) -> str:
        """Convert byte sequence to human readable string representation"""
        try:
            # Try to decode as UTF-8 string
            return byte_seq.decode('utf-8')
        except UnicodeDecodeError:
            # If decoding fails, return hexadecimal representation
            return '0x' + byte_seq.hex()

    def save_tokenizer(self, prefix: str = "tokenizer") -> None:
        """
        Save tokenizer vocabulary and merge rules
        Args:
            prefix: The prefix for the save file
        """
        # Prepare vocabulary data - convert to readable format
        readable_vocab = {}
        for k, v in self._base_tokenizer.vocab.items():
            readable_vocab[str(k)] = {
                'bytes': list(v),  # Save original byte list for restoration
                'readable': self._bytes_to_readable(v)  # Human readable string representation
            }

        # Prepare merge rules data - convert to readable format
        readable_merges = {}
        for (id1, id2), merged_id in self._base_tokenizer.merges.items():
            # Get original byte sequences
            token1 = self._base_tokenizer.vocab[id1]
            token2 = self._base_tokenizer.vocab[id2]
            merged_token = self._base_tokenizer.vocab[merged_id]
            
            # Create readable representation
            readable_merges[f"{merged_id}"] = {
                'components': [
                    {'id': id1, 'token': self._bytes_to_readable(token1)},
                    {'id': id2, 'token': self._bytes_to_readable(token2)}
                ],
                'merged': self._bytes_to_readable(merged_token),
                'original_pair': [id1, id2]  # Save original ID pair for restoration
            }

        # Prepare complete tokenizer data
        tokenizer_data = {
            "special_tokens": self._special_tokens,
            "chain_break_token": self._cb_token,
            "vocab_size": self._total_vocab_size,
            "base_vocab_size": self._base_vocab_size,
            "special_token_ids": self._special_token_ids,
            "base_tokenizer": {
                "vocab": readable_vocab,
                "merges": readable_merges
            }
        }
        
        # Save in JSON format
        save_path = os.path.join(self.checkpoint_dir, f"{prefix}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)
        
        print(f"Tokenizer saved to {save_path}")

    def load_tokenizer(self, prefix: str = "tokenizer") -> None:
        """
        Load saved tokenizer configuration
        Args:
            prefix: The prefix of the file to load
        """
        load_path = os.path.join(self.checkpoint_dir, f"{prefix}.json")
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No tokenizer file found at {load_path}")
            
        with open(load_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
            
        # Restore special tokens
        self._special_tokens = tokenizer_data["special_tokens"]
        self._cb_token = tokenizer_data["chain_break_token"]
        self._total_vocab_size = tokenizer_data["vocab_size"]
        self._base_vocab_size = tokenizer_data["base_vocab_size"]
        self._special_token_ids = tokenizer_data["special_token_ids"]
        
        # Restore vocabulary
        self._base_tokenizer.vocab = {
            int(k): bytes(v['bytes']) 
            for k, v in tokenizer_data["base_tokenizer"]["vocab"].items()
        }
        
        # Restore merge rules
        self._base_tokenizer.merges = {
            tuple(merge_info['original_pair']): int(merge_id)
            for merge_id, merge_info in tokenizer_data["base_tokenizer"]["merges"].items()
        }
        
        print(f"Tokenizer loaded from {load_path}")