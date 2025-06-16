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