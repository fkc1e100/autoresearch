import sys
import os
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# Add the current directory to the path so we can import train
sys.path.append(os.path.dirname(__file__))

from train import GPT, GPTConfig
from prepare import Tokenizer

class InferenceServer(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            prompt = data.get('prompt', '')
            max_tokens = data.get('max_tokens', 50)
            
            if not prompt:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'Missing prompt')
                return
                
            print(f"Generating for prompt: {prompt}")
            
            # Encode prompt
            tokens = tokenizer.encode(prompt)
            x = torch.tensor([tokens], dtype=torch.long, device=device)
            
            # Generate
            model.eval()
            with torch.no_grad():
                with autocast_ctx:
                    for _ in range(max_tokens):
                        logits = model(x)
                        logits = logits[:, -1, :] # get last token
                        probs = torch.softmax(logits, dim=-1)
                        # Sample or argmax. Let's do greedy for simplicity first
                        next_token = torch.argmax(probs, dim=-1, keepdim=True)
                        x = torch.cat((x, next_token), dim=1)
                        
            # Decode
            output_tokens = x[0].tolist()
            response_text = tokenizer.decode(output_tokens)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {'text': response_text}
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def load_model():
    global model, tokenizer, device, autocast_ctx
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_ctx = torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)
    
    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    
    from train import DEPTH, build_model_config
    config = build_model_config(DEPTH)
    
    model = GPT(config)
    
    save_path = "/workspace/training_data/model.pt"
    if not os.path.exists(save_path):
        print(f"Model file not found at {save_path}. Exiting.")
        sys.exit(1)
        
    print(f"Loading model from {save_path}...")
    state_dict = torch.load(save_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print("Model loaded successfully.")

if __name__ == '__main__':
    load_model()
    server_address = ('', 8080)
    httpd = HTTPServer(server_address, InferenceServer)
    print('Starting inference server on port 8080...')
    httpd.serve_forever()
