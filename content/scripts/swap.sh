# !/bin/bash

cd ../SS/SimSwap
python test_one_image.py --name people --crop_size 224 --Arc_path ./arcface_model/arcface_checkpoint.tar --pic_a_path $1 --pic_b_path $2 --output_path ../../SwpTmp/
cd ../../scripts