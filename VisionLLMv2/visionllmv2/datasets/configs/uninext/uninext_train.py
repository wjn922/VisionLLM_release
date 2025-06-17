datasets = [
    # chat
    {
        'type': 'llava_data',
        'ann_file': './data/sharegpt4v/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json',
        'img_prefix': './data/',
        'ratio': 0.05  # ori 6665058
    },
    {
        'type': 'llava_data',
        'ann_file': 'data/sft_data/ALLaVA-Caption-LAION-4V.jsonl',
        'img_prefix': 'data/',
        'ratio': 0.03 # ori 457130
    },
    {
        'type': 'llava_data',
        'ann_file': 'data/sft_data/ALLaVA-Instruct-LAION-4V.jsonl',
        'img_prefix': 'data/',
        'ratio': 0.03 # ori 457130
    },
    # od
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_od_coco_5k.jsonl',
        'img_prefix': './datasets/coco/train2017',
    },
    # is
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_is_coco_5k.jsonl',
        'img_prefix': './datasets/coco/train2017',
    },
    # rec
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_rec_refcoco_5k.jsonl',
        'img_prefix': './datasets/coco/train2014',
    },
    # res
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_res_refcoco_5k.jsonl',
        'img_prefix': './datasets/coco/train2014',
    },
    # vis
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_vis_ytvis19_5k.jsonl',
        'img_prefix': './datasets/ytvis_2019/train/JPEGImages',
    },
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_vis_ovis_5k.jsonl',
        'img_prefix': './datasets/ovis/train',
    },
    # vos
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_vos_ytbvos18_5k.jsonl',
        'img_prefix': './datasets/ytbvos18/train/JPEGImages',
    },
    # rvos
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_rvos_refytvos_5k.jsonl',
        'img_prefix': './datasets/ref-youtube-vos/train/JPEGImages',
    },
    # sot
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_sot_lasot_5k.jsonl',
        'img_prefix': './datasets/LaSOT',
    },
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_sot_trackingnet_5k.jsonl',
        'img_prefix': './datasets/TrackingNet',
    },
    # mot
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_mot_bdd_5k.jsonl',
        'img_prefix': './datasets/bdd/images/track/train',
    },
    # mots
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_mots_bdd_5k.jsonl',
        'img_prefix': './datasets/bdd/images/seg_track_20/train',
    },
]