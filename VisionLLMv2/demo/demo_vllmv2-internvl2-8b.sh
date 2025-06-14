# visionllmv2-llava-7b
s1a python3 -m demo.run_vllmv2_uninext \
    --model-name work_dirs/internvl2-8b \
    --image-file assets/coco2.jpg \
    --query "Please brielfy describe the image content." \
    --conv-mode internlm2_chat \
    --image_aspect_ratio anyres \
    --image_size 448 \
    --image_max_tile 6 \
    --use_pixelshuffle True