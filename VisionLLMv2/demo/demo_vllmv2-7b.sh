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

# -------------------------------------------------------------
# rvos
s1a python3 -m demo.run_vllmv2_uninext \
    --model-name work_dirs/visionllmv2-7b-pure-ft-uninext \
    --image-folder datasets/ref-youtube-vos/valid/JPEGImages/3dd327ab4e \
    --query "Can you create frame-by-frame segmentation masks for the brown cow in the middle shown in the video?" \
    --conv-mode vicuna_v1 \
    --image_aspect_ratio anyres \
    --image_size 336 \
    --image_max_tile 4 

# -------------------------------------------------------------
# vos
s1a python3 -m demo.run_vllmv2_uninext \
    --model-name work_dirs/visionllmv2-7b-pure-ft-uninext \
    --image-folder datasets/ytbvos18/val/JPEGImages/1ecc34b1bf \
    --ref_mask assets/ytbvos18_1ecc34b1bf_ref.png \
    --query "Can you segment and track the moving object in the video using the initial mask provided?" \
    --conv-mode vicuna_v1 \
    --image_aspect_ratio anyres \
    --image_size 336 \
    --image_max_tile 4 

# -------------------------------------------------------------
# sot
s1a python3 -m demo.run_vllmv2_uninext \
    --model-name work_dirs/visionllmv2-7b-pure-ft-uninext \
    --image-folder datasets/LaSOT/basketball/basketball-1/img \
    --ref_box "[347, 135, 366, 153]" \
    --query "Use the initial box annotation to generate a tracking trajectory for the single target." \
    --conv-mode vicuna_v1 \
    --image_aspect_ratio anyres \
    --image_size 336 \
    --image_max_tile 4 

# -------------------------------------------------------------
# mot / mots
s1a python3 -m demo.run_vllmv2_uninext \
    --model-name work_dirs/visionllmv2-7b-pure-ft-uninext \
    --image-folder datasets/bdd/images/seg_track_20/val/b1c9c847-3bda4659 \
    --query "Could you predict bounding boxes and segmentation masks for all car instances in the video sequence?" \
    --conv-mode vicuna_v1 \
    --image_aspect_ratio anyres \
    --image_size 336 \
    --image_max_tile 4 