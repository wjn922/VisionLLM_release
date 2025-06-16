# visionllmv2-llava-7b
# for vision tasks, outputs are saved to 'uninext_outputs'

# -------------------------------------------------------------
# captioning
s1a python3 -m demo.run_vllmv2_uninext \
    --model-name work_dirs/visionllmv2-internvl2-8b-ft-uninext \
    --image-file assets/coco2.jpg \
    --query "Please brielfy describe the image content." \
    --conv-mode internlm2_chat \
    --image_aspect_ratio anyres \
    --image_size 448 \
    --image_max_tile 6 \
    --use_pixelshuffle True

    # -------------------------------------------------------------
# od/is
s1a python3 -m demo.run_vllmv2_uninext \
    --model-name work_dirs/visionllmv2-internvl2-8b-ft-uninext \
    --image-file assets/coco2.jpg \
    --query "Please detect and mark all instances of person, tennis racket within the photograph." \
    --conv-mode internlm2_chat \
    --image_aspect_ratio anyres \
    --image_size 448 \
    --image_max_tile 6 \
    --use_pixelshuffle True

    # -------------------------------------------------------------
# rec/res
s1a python3 -m demo.run_vllmv2_uninext \
    --model-name work_dirs/visionllmv2-internvl2-8b-ft-uninext \
    --image-file assets/coco3.jpg \
    --query "Could you find the the leftmost sheep in the image?" \
    --conv-mode internlm2_chat \
    --image_aspect_ratio anyres \
    --image_size 448 \
    --image_max_tile 6 \
    --use_pixelshuffle True

# -------------------------------------------------------------
# vis
s1a python3 -m demo.run_vllmv2_uninext \
    --model-name work_dirs/visionllmv2-internvl2-8b-ft-uninext \
    --image-folder datasets/ytvis_2019/val/JPEGImages/03a2bade84 \
    --query "Please segment and track all instances of deer, car in the video with detailed masks." \
    --conv-mode internlm2_chat \
    --image_aspect_ratio anyres \
    --image_size 448 \
    --image_max_tile 6 \
    --use_pixelshuffle True

# -------------------------------------------------------------
# rvos
s1a python3 -m demo.run_vllmv2_uninext \
    --model-name work_dirs/visionllmv2-internvl2-8b-ft-uninext \
    --image-folder datasets/ref-youtube-vos/valid/JPEGImages/3dd327ab4e \
    --query "Could you provide the segmentation masks of the brown cow for every single frame of the video?" \
    --conv-mode internlm2_chat \
    --image_aspect_ratio anyres \
    --image_size 448 \
    --image_max_tile 6 \
    --use_pixelshuffle True

# -------------------------------------------------------------
# vos
s1a python3 -m demo.run_vllmv2_uninext \
    --model-name work_dirs/visionllmv2-internvl2-8b-ft-uninext \
    --image-folder datasets/ytbvos18/val/JPEGImages/1ecc34b1bf \
    --ref_mask assets/ytbvos18_1ecc34b1bf_ref.png \
    --query "Can you segment and track the moving object in the video using the initial mask provided?" \
    --conv-mode internlm2_chat \
    --image_aspect_ratio anyres \
    --image_size 448 \
    --image_max_tile 6 \
    --use_pixelshuffle True

# -------------------------------------------------------------
# sot
s1a python3 -m demo.run_vllmv2_uninext \
    --model-name work_dirs/visionllmv2-internvl2-8b-ft-uninext \
    --image-folder datasets/LaSOT/basketball/basketball-1/img \
    --ref_box "[347, 135, 366, 153]" \
    --query "Use the initial box annotation to generate a tracking trajectory for the single target." \
    --conv-mode internlm2_chat \
    --image_aspect_ratio anyres \
    --image_size 448 \
    --image_max_tile 6 \
    --use_pixelshuffle True

# -------------------------------------------------------------
# mot / mots
s1a python3 -m demo.run_vllmv2_uninext \
    --model-name work_dirs/visionllmv2-internvl2-8b-ft-uninext \
    --image-folder datasets/bdd/images/seg_track_20/val/b1c9c847-3bda4659 \
    --query "Could you predict bounding boxes and segmentation masks for all car instances in the video sequence?" \
    --conv-mode internlm2_chat \
    --image_aspect_ratio anyres \
    --image_size 448 \
    --image_max_tile 6 \
    --use_pixelshuffle True