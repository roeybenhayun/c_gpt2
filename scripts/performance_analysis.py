import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# --- Configuration ---

# To use MANUAL mode: Fill in the paths for the files you want to plot.
# To use AUTOMATIC mode: Leave all paths as empty strings ("").
files = {
    "gpt2_small": "",
    "gpt2_medium": "",
    "gpt2_large": ""
}

LOG_DIR = "logs"
files_to_process = {}

# --- Mode Selection ---

is_manual_mode = any(files.values())

if is_manual_mode:
    print("--- Running in MANUAL mode ---")
    files_to_process = {model: path for model, path in files.items() if path}
    for model, path in files_to_process.items():
        print(f"  -> Using manual path for '{model}': {path}")

else:
    print("--- Running in AUTOMATIC discovery mode ---")
    for model_name in files.keys():
        search_pattern = os.path.join(LOG_DIR, f"{model_name}_*.json")
        found_files = glob.glob(search_pattern)
        
        if not found_files:
            print(f"⚠️  Warning: No result file found for model '{model_name}'. Skipping.")
            continue
            
        latest_file = max(found_files, key=os.path.getmtime)
        files_to_process[model_name] = latest_file
        print(f"  -> Found latest for '{model_name}': {os.path.basename(latest_file)}")

# --- Final Check & Plotting ---

if not files_to_process:
    raise FileNotFoundError("No files to process. Please fill in paths manually or run the bash script to generate log files.")

print("\n--- Processing and plotting data ---")

plt.figure(figsize=(10, 6))
chunk_size_for_label = None

for model_name, file_path in files_to_process.items():
    try:
        with open(file_path) as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing {file_path}: {e}")
        continue

    if chunk_size_for_label is None:
        chunk_size_for_label = data.get("token_chunk_size", "N/A")

    # --- Corrected logic for setting fit order ---
    kv_cache_enabled = data.get("kv_cache_enabled", 1) # Default to 1 (enabled)
    
    if kv_cache_enabled == 1:
        order = 1  # Linear fit for KV cache enabled
    else:
        order = 2  # Quadratic fit for KV cache disabled

    context_lengths = [chunk["total_context"] for chunk in data["chunks"]]
    chunk_times = [chunk["chunk_seconds"] for chunk in data["chunks"]]

    x = np.array(context_lengths)
    y = np.array(chunk_times)
    
    coeffs = np.polyfit(x, y, order)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = poly(x_fit)

    plt.plot(x, y, 'o', label=f'{model_name} data')
    plt.plot(x_fit, y_fit, '--', label=f'{model_name} fit (order {order})')

plt.xlabel("Total Context Length")
plt.ylabel(f"Time for Last {chunk_size_for_label} Tokens (s)")
plt.title("Time to Generate Tokens vs. Context Length")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()