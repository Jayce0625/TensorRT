import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'

# default cache
default_home = os.path.join(os.path.expanduser("~"), ".cache")
hf_cache_home = os.path.expanduser(
    os.getenv(
        "HF_HOME",
        os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "huggingface"),
    )
)
default_cache_path = os.path.join(hf_cache_home, "hub")
HUGGINGFACE_HUB_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE", default_cache_path)

import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description='Download a specified model/dataset from HuggingFace.')
parser.add_argument("-r", "--repo_id", help="HF model/dataset repo id.", required=True)
parser.add_argument("-t", "--repo_type", help="HF repo type. It can be model or dataset", default=None)
parser.add_argument("-c", "--cache_dir", help="The path where you want to store the model/dataset's source files.", default=HUGGINGFACE_HUB_CACHE)
parser.add_argument("-l", "--local_dir", help="The path where you want to store the model/dataset & link.", default="./hf_models/")
parser.add_argument("-f", "--flag", help="The flag determines whether or not to use symlinks to store in local_dir. E.g. \"auto\" or True or False.", default="auto")
args = parser.parse_args()
# print(args.repo_id)

if not os.path.exists(args.cache_dir):
    os.makedirs(args.cache_dir)
    print(f"缓存模型文件夹已成功创建: {args.cache_dir}")
else:
    print(f"缓存模型文件夹已存在: {args.cache_dir}")

if args.local_dir[-1] != "/":
    args.local_dir += "/"
args.local_dir += args.repo_id

if not os.path.exists(args.local_dir):
    os.makedirs(args.local_dir)
    print(f"本地模型文件夹已成功创建: {args.local_dir}")
else:
    print(f"本地模型文件夹已存在: {args.local_dir}")

snapshot_download(
    repo_id=args.repo_id,
    repo_type=args.repo_type,
    cache_dir=args.cache_dir,
    local_dir=args.local_dir,
    local_dir_use_symlinks=args.flag,
    resume_download=True,
    allow_patterns=["*.model","*.json","*.bin","*.py","*.md","*.txt","*.tar","*.safetensors","*.ckpt"],
    ignore_patterns=["*.msgpack","*.ot","*.h5","*non_ema*","*pruned.*"],
    max_workers=8
)