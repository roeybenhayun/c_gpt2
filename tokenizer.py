# tokenizer_server.py
import socket
import json
from tokenizers import Tokenizer

# Load the tokenizer from a local tokenizer.json file
tokenizer = Tokenizer.from_file("../gpt2/tokenizer.json")

HOST = '127.0.0.1'
PORT = 65432

def handle_request(data):
    try:
        req = json.loads(data)
        mode = req.get("mode")
        if mode == "encode":
            print("Encode request received...")
            text = req.get("text", "")
            tokens = tokenizer.encode(text).ids
            return json.dumps({"tokens": tokens})
        elif mode == "decode":
            print("Decode request received...")
            tokens = req.get("tokens", [])
            print(tokens)
            text = tokenizer.decode(tokens)
            return json.dumps({"text": text})
        else:
            return json.dumps({"error": "Invalid mode"})
    except Exception as e:
        return json.dumps({"error": str(e)})

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Tokenizer server listening on {HOST}:{PORT}...")
    while True:
        conn, addr = s.accept()
        with conn:
            data = conn.recv(4096).decode()
            response = handle_request(data)
            conn.sendall(response.encode())
