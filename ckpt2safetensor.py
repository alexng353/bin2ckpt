import torch
from safetensors.torch import save_file
import os

# traverse directory C:\Users\alexn\Desktop\stable\models one depth and find all ckpt files

files = os.listdir("C:\\Users\\alexn\\Desktop\\stable\\models")
files = [f for f in files if f.endswith(".ckpt")]

done = os.listdir("C:\\Users\\alexn\\Desktop\\stable\\models\\safetensors")

files = [f for f in files if f.split(
    ".")[0] not in [d.split(".")[0] for d in done]]

skip = ["HD-16.ckpt", "HD-17.ckpt",
        "sd-v1-5-fp16.ckpt", "v1-5-pruned-emaonly.ckpt"]

for f in files:
    print(f)
    if f in skip:
        print("skipping")
        continue
    weights = torch.load(f"C:\\Users\\alexn\\Desktop\\stable\\models\\{f}")[
        "state_dict"]
    save_file(
        weights, f"C:\\Users\\alexn\\Desktop\\stable\\models\\safetensors\\{f.split('.')[0]}.safetensors")
    print("done")
