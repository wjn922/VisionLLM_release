# od - coco
python3 tools_vllmv2/convert_od_to_conv.py --input_path datasets/coco/annotations/instances_train2017.json --output_path datasets/uninext_sft_data/conversation_od_coco_4k.jsonl --samples 4000

# is - coco
python3 tools_vllmv2/convert_is_to_conv.py --input_path datasets/coco/annotations/instances_train2017.json --output_path datasets/uninext_sft_data/conversation_is_coco_4k.jsonl --samples 4000

# rec - refcoco
python3 tools_vllmv2/convert_rec_to_conv.py --input_path datasets/annotations/refcoco-mixed/instances_train.json --output_path datasets/uninext_sft_data/conversation_rec_refcoco_4k.jsonl --samples 4000

# res - refcoco
python3 tools_vllmv2/convert_res_to_conv.py --input_path datasets/annotations/refcoco-mixed/instances_train.json --output_path datasets/uninext_sft_data/conversation_res_refcoco_4k.jsonl --samples 4000

# vis - ytvis19
python3 tools_vllmv2/convert_vis_to_conv.py --input_path datasets/ytvis_2019/annotations/train.json --output_path datasets/uninext_sft_data/conversation_vis_ytvis19_4k.jsonl --samples 4000

# vis - ovis
python3 tools_vllmv2/convert_vis_to_conv.py --input_path datasets/ovis/annotations_train.json --output_path datasets/uninext_sft_data/conversation_vis_ovis_4k.jsonl --samples 4000

# vos - ytbvos18
python3 tools_vllmv2/convert_vos_to_conv.py --input_path datasets/ytbvos18/train/train.json --output_path datasets/uninext_sft_data/conversation_vos_ytbvos18_4k.jsonl --samples 4000

# rvos - refytvos
python3 tools_vllmv2/convert_rvos_to_conv.py --input_path datasets/ref-youtube-vos/train.json --output_path datasets/uninext_sft_data/conversation_rvos_refytvos_4k.jsonl --samples 4000

# sot - lasot
python3 tools_vllmv2/convert_sot_to_conv.py --input_path datasets/LaSOT/test.json --output_path datasets/uninext_sft_data/conversation_sot_lasot_4k.jsonl --samples 4000

# sot - trackingnet
python3 tools_vllmv2/convert_sot_to_conv.py --input_path datasets/TrackingNet/TEST.json --output_path datasets/uninext_sft_data/conversation_sot_trackingnet_4k.jsonl --samples 4000

# mot - bdd
python3 tools_vllmv2/convert_mot_to_conv.py --input_path datasets/bdd/labels/box_track_20/box_track_train_cocofmt_uni.json --output_path datasets/uninext_sft_data/conversation_mot_bdd_4k.jsonl --samples 4000

# mots - bdd
python3 tools_vllmv2/convert_mots_to_conv.py --input_path datasets/bdd/labels/seg_track_20/seg_track_train_cocoformat_uni.json --output_path datasets/uninext_sft_data/conversation_mots_bdd_4k.jsonl --samples 4000