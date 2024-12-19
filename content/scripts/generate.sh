# !/bin/bash
cd ../SD/stable-diffusion
input_string="$1"
python scripts/txt2img.py --prompt "$input_string" --plms --outdir ../../GenTmp
cd ../../scripts
python crop.py

