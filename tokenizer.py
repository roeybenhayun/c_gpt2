# tokenizer_server.py
import socket
import json
from tokenizers import Tokenizer

# Load the tokenizer from a local tokenizer.json file
# https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json
tokenizer = Tokenizer.from_file("tokenizer.json")

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


# Read in a loop until the request parses as valid JSON. Single recv(4096) was
# truncating long encode requests (~5 KB+ for ~1000-token prompts) and producing
# JSON-decode errors that surfaced on the client as "Failed to parse tokens".
MAX_REQUEST_BYTES = 64 * 1024

def recv_full_json(conn):
    buf = b""
    while len(buf) < MAX_REQUEST_BYTES:
        chunk = conn.recv(4096)
        if not chunk:
            break  # peer closed
        buf += chunk
        try:
            json.loads(buf.decode("utf-8"))
            return buf.decode("utf-8")
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue  # need more bytes
    return buf.decode("utf-8", errors="replace")


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Tokenizer server listening on {HOST}:{PORT}...")
    while True:
        conn, addr = s.accept()
        with conn:
            data = recv_full_json(conn)
            response = handle_request(data)
            conn.sendall(response.encode())
