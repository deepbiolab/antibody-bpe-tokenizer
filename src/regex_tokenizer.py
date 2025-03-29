import regex as re
from typing import List
from .basic_tokenizer import BasicTokenizer

class RegexTokenizer(BasicTokenizer):
    def __init__(self):
        super().__init__()
        self.pattern = re.compile(r""" ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
    def encode(self, text: str) -> List[int]:
        """Encode text using regex pattern first"""
        parts = re.findall(self.pattern, text)
        all_tokens = []
        for part in parts:
            tokens = super().encode(part)
            all_tokens.extend(tokens)
        return all_tokens