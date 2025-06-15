# visionllmv2-llava-7b
# for vision tasks, outputs are saved to 'uninext_outputs'

# -------------------------------------------------------------
# captioning
s1a python3 -m demo.run_vllmv2_uninext \
    --model-name work_dirs/visionllmv2-7b-pure-ft-uninext \
    --image-file assets/coco2.jpg \
    --query "Please brielfy describe the image content." \
    --conv-mode vicuna_v1 \
    --image_aspect_ratio anyres \
    --image_size 336 \
    --image_max_tile 4 

# -------------------------------------------------------------
# od/is
s1a python3 -m demo.run_vllmv2_uninext \
    --model-name work_dirs/visionllmv2-7b-pure-ft-uninext \
    --image-file assets/coco2.jpg \
    --query "Please detect and mark all instances of person, tennis racket within the photograph." \
    --conv-mode vicuna_v1 \
    --image_aspect_ratio anyres \
    --image_size 336 \
    --image_max_tile 4 

# -------------------------------------------------------------
# rec/res
s1a python3 -m demo.run_vllmv2_uninext \
    --model-name work_dirs/visionllmv2-7b-pure-ft-uninext \
    --image-file assets/coco3.jpg \
    --query "Could you find the the leftmost sheep in the image?" \
    --conv-mode vicuna_v1 \
    --image_aspect_ratio anyres \
    --image_size 336 \
    --image_max_tile 4 

# -------------------------------------------------------------
# vis
s1a python3 -m demo.run_vllmv2_uninext \
    --model-name work_dirs/visionllmv2-7b-pure-ft-uninext \
    --image-folder datasets/ytvis_2019/val/JPEGImages/03a2bade84 \
    --query "Please segment and track all instances of deer, car in the video with detailed masks." \
    --conv-mode vicuna_v1 \
    --image_aspect_ratio anyres \
    --image_size 336 \
    --image_max_tile 4 