datasets = [
    # od
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_od_coco_3k.jsonl',
        'img_prefix': './datasets/coco/train2017',
    },
    # is
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_is_coco_3k.jsonl',
        'img_prefix': './datasets/coco/train2017',
    },
    # rec
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_rec_refcoco_3k.jsonl',
        'img_prefix': './datasets/coco/train2014',
    },
    # res
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_res_refcoco_3k.jsonl',
        'img_prefix': './datasets/coco/train2014',
    },
    # vis
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_vis_ytvis19_3k.jsonl',
        'img_prefix': './datasets/ytvis_2019/train/JPEGImages',
    },
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_vis_ovis_3k.jsonl',
        'img_prefix': './datasets/ovis/train',
    },
    # vos
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_vos_ytbvos18_3k.jsonl',
        'img_prefix': './datasets/ytbvos18/train/JPEGImages',
    },
    # rvos
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_rvos_refytvos_3k.jsonl',
        'img_prefix': './datasets/ref-youtube-vos/train/JPEGImages',
    },
    # sot
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_sot_lasot_3k.jsonl',
        'img_prefix': './datasets/LaSOT',
    },
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_sot_trackingnet_3k.jsonl',
        'img_prefix': './datasets/TrackingNet',
    },
    # mot
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_mot_bdd_3k.jsonl',
        'img_prefix': './datasets/bdd/images/track/train',
    },
    # mots
    {
        'type': 'llava_data',
        'ann_file': './datasets/uninext_sft_data/conversation_mots_bdd_3k.jsonl',
        'img_prefix': './datasets/bdd/images/seg_track_20/train',
    },
]