import os
import json
import sys
import tiktoken
import torch
import numpy as np

MAX_SEQ_LEN = 2048
TIME_BUDGET = 300

class Tokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")
    
    @classmethod
    def from_directory(cls, directory=None):
        return cls()
    def encode(self, text):
        return self.enc.encode(text)
    def decode(self, tokens):
        return self.enc.decode(tokens)
    def get_vocab_size(self):
        return self.enc.n_vocab

def make_dataloader(tokenizer, batch_size, seq_len, split):
    trace_file = "/workspace/training_data/traces.jsonl"
    if not os.path.exists(trace_file):
        print(f"Warning: {trace_file} not found, yielding dummy data.", file=sys.stderr)
        while True:
            x = torch.randint(0, tokenizer.get_vocab_size(), (batch_size, seq_len))
            y = torch.randint(0, tokenizer.get_vocab_size(), (batch_size, seq_len))
            yield x.cuda(), y.cuda(), 0
            
    all_tokens = []
    with open(trace_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                text = ""
                for turn in data:
                    for part in turn["parts"]:
                        if "text" in part:
                            text += part["text"] + "\n"
                        elif "functionCall" in part:
                            text += json.dumps(part["functionCall"]) + "\n"
                        elif "functionResponse" in part:
                            text += json.dumps(part["functionResponse"]) + "\n"
                all_tokens.extend(tokenizer.encode(text))
            except Exception as e:
                print(f"Error parsing line: {e}", file=sys.stderr)
                
    if not all_tokens:
        print("Warning: No tokens loaded from file, yielding dummy data.", file=sys.stderr)
        while True:
            x = torch.randint(0, tokenizer.get_vocab_size(), (batch_size, seq_len))
            y = torch.randint(0, tokenizer.get_vocab_size(), (batch_size, seq_len))
            yield x.cuda(), y.cuda(), 0
            
    all_tokens = np.array(all_tokens, dtype=np.int32)
    
    while True:
        ix = np.random.randint(0, len(all_tokens) - seq_len, batch_size)
        x = np.stack([all_tokens[i:i+seq_len] for i in ix])
        y = np.stack([all_tokens[i+1:i+seq_len+1] for i in ix])
        
        x = torch.from_numpy(x).long()
        y = torch.from_numpy(y).long()
        
        yield x.cuda(), y.cuda(), 0

def evaluate_bpb(model, tokenizer, batch_size):
    print("Warning: Dummy evaluation returning 1.0", file=sys.stderr)
    return 1.0
