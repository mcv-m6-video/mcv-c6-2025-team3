from huggingface_hub import snapshot_download
snapshot_download(repo_id="SoccerNet/SN-BAS-2025",
repo_type="dataset", revision="main",
local_dir="/home/danielpardo/c6/SoccerNet/SN-BAS-2025")