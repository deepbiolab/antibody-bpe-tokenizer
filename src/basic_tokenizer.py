from typing import List, Dict, Tuple

class BasicTokenizer:
    def __init__(self):
        self.merges = {}  # Dictionary to store merge rules
        self.vocab = {}   # Dictionary to store vocabulary
        self.vocab_size = 0
        
    def get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        """Count frequency of adjacent token pairs"""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        """Merge adjacent tokens according to learned rules"""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train the tokenizer on input text"""
        # Initialize base vocabulary with byte tokens
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.vocab_size = vocab_size
        
        # Convert text to initial byte tokens
        tokens = list(text.encode('utf-8'))
        ids = list(tokens)
        
        # Perform merges until desired vocabulary size is reached
        num_merges = vocab_size - 256
        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
                
            max_pair = max(stats, key=stats.get)
            idx = 256 + i
            
            if verbose:
                print(f"merging {max_pair} into a new id {idx}")
                
            ids = self.merge(ids, max_pair, idx)
            self.merges[max_pair] = idx
            self.vocab[idx] = self.vocab[max_pair[0]] + self.vocab[max_pair[1]]
            
    def encode(self, text: str) -> List[int]:
        """Encode text into token ids"""
        tokens = list(text.encode('utf-8'))
        while len(tokens) > 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            tokens = self.merge(tokens, pair, self.merges[pair])
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """Decode token ids back to text"""
        return b''.join(self.vocab[idx] for idx in ids).decode('utf-8')