# visionllmv2-llava-7b
# for vision tasks, outputs are saved to 'uninext_outputs'
s1a python3 -m demo.run_vllmv2_uninext \
    --model-name work_dirs/visionllmv2-7b-pure \
    --image-file assets/coco2.jpg \
    --query "Please brielfy describe the image content." \
    --conv-mode vicuna_v1 \
    --image_aspect_ratio anyres \
    --image_size 336 \
    --image_max_tile 4 