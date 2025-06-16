# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from typing import Dict, List
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from .backbone.masked_backbone import MaskedBackbone
from .models.deformable_detr.backbone import Joiner
from .models.deformable_detr.deformable_detr import DeformableDETR, SetCriterion, DeformableDETRDINO, DINOCriterion
from .models.deformable_detr.matcher import HungarianMatcherVL
from .models.deformable_detr.position_encoding import PositionEmbeddingSine
from .models.deformable_detr.deformable_transformer import DeformableTransformerVL
from .models.deformable_detr.deformable_transformer_dino import DeformableTransformerVLDINO
from .models.ddetrs_vid import DDETRSegmUniVID
from .models.ddetrs_vid_dn import DDETRSegmUniVIDDN
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import NestedTensor
import torchvision.ops as ops
# Language-guided detection
from transformers import AutoTokenizer
from .models.deformable_detr.bert_model import BertEncoder

from collections import OrderedDict
from einops import repeat
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)
from detectron2.structures import BoxMode
import cv2
from .models.tracker import IDOL_Tracker, QuasiDenseEmbedTracker
import copy
from collections import defaultdict
from PIL import Image
import time
from detectron2.layers import ShapeSpec
import pycocotools.mask as mask_util
from scipy.optimize import linear_sum_assignment
__all__ = ["UNINEXT_VID_DEMO"]
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

@META_ARCH_REGISTRY.register()
class UNINEXT_VID_DEMO(nn.Module):
    """
    Unified model for video-level tasks (SOT, VOS, R-VOS, MOT, MOTS, VIS)
    """

    def __init__(self, cfg):
        super().__init__()
        self.debug_only = False
        self.cfg = cfg
        self.use_amp = cfg.SOLVER.AMP.ENABLED
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.mask_stride = cfg.MODEL.DDETRS.MASK_STRIDE
        self.mask_on = cfg.MODEL.MASK_ON
        self.ota = cfg.MODEL.OTA
        self.mask_thres = cfg.MODEL.DDETRS.MASK_THRES
        self.new_mask_head = cfg.MODEL.DDETRS.NEW_MASK_HEAD
        self.use_raft = cfg.MODEL.DDETRS.USE_RAFT
        self.use_rel_coord = cfg.MODEL.DDETRS.USE_REL_COORD
        self.num_queries = cfg.MODEL.DDETRS.NUM_OBJECT_QUERIES

        ### inference setting
        self.merge_on_cpu = cfg.MODEL.IDOL.MERGE_ON_CPU
        self.is_multi_cls = cfg.MODEL.IDOL.MULTI_CLS_ON
        self.apply_cls_thres = cfg.MODEL.IDOL.APPLY_CLS_THRES
        self.temporal_score_type = cfg.MODEL.IDOL.TEMPORAL_SCORE_TYPE
        self.inference_select_thres = cfg.MODEL.IDOL.INFERENCE_SELECT_THRES
        self.inference_fw = cfg.MODEL.IDOL.INFERENCE_FW
        self.inference_tw = cfg.MODEL.IDOL.INFERENCE_TW
        self.memory_len = cfg.MODEL.IDOL.MEMORY_LEN
        self.batch_infer_len = cfg.MODEL.IDOL.BATCH_INFER_LEN # 10
        self.merge_device = "cpu" if self.merge_on_cpu else self.device
        self.save_path_prefix = os.path.join(cfg.OUTPUT_DIR, "inference/refytvos")

        # Transformer parameters:
        hidden_dim = cfg.MODEL.DDETRS.HIDDEN_DIM
        nheads = cfg.MODEL.DDETRS.NHEADS
        dim_feedforward = cfg.MODEL.DDETRS.DIM_FEEDFORWARD
        dec_layers = cfg.MODEL.DDETRS.DEC_LAYERS

        num_feature_levels = cfg.MODEL.DDETRS.NUM_FEATURE_LEVELS
        two_stage = cfg.MODEL.DDETRS.TWO_STAGE
        two_stage_num_proposals = cfg.MODEL.DDETRS.TWO_STAGE_NUM_PROPOSALS

        # Loss parameters:
        mask_weight = cfg.MODEL.DDETRS.MASK_WEIGHT
        dice_weight = cfg.MODEL.DDETRS.DICE_WEIGHT
        giou_weight = cfg.MODEL.DDETRS.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DDETRS.L1_WEIGHT
        class_weight = cfg.MODEL.DDETRS.CLASS_WEIGHT
        reid_weight = cfg.MODEL.DDETRS.REID_WEIGHT
        deep_supervision = cfg.MODEL.DDETRS.DEEP_SUPERVISION
        focal_alpha = cfg.MODEL.DDETRS.FOCAL_ALPHA
        # Cost parameters (for label assignment):
        set_cost_class = cfg.MODEL.DDETRS.SET_COST_CLASS
        set_cost_bbox = cfg.MODEL.DDETRS.SET_COST_BOX
        set_cost_giou = cfg.MODEL.DDETRS.SET_COST_GIOU

        # Backbone
        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels  # only take [c3 c4 c5] from resnet and gengrate c6 later
        backbone.strides = d2_backbone.feature_strides

        # Transformer & Early Fusion
        if cfg.MODEL.DDETRS.USE_DINO:
            transformer_class = DeformableTransformerVLDINO
        else:
            transformer_class = DeformableTransformerVL
        transformer = transformer_class(
        d_model= hidden_dim,
        nhead=nheads,
        num_encoder_layers=cfg.MODEL.DDETRS.ENC_LAYERS,
        num_decoder_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        dropout=cfg.MODEL.DDETRS.DROPOUT,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=num_feature_levels,
        dec_n_points=cfg.MODEL.DDETRS.DEC_N_POINTS,
        enc_n_points=cfg.MODEL.DDETRS.ENC_N_POINTS,
        two_stage=two_stage,
        two_stage_num_proposals=two_stage_num_proposals,
        use_checkpoint=cfg.MODEL.DDETRS.USE_CHECKPOINT,
        look_forward_twice=cfg.MODEL.DDETRS.LOOK_FORWARD_TWICE,
        mixed_selection=cfg.MODEL.DDETRS.MIXED_SELECTION,
        cfg=cfg)
        
        if cfg.MODEL.DDETRS.USE_DINO:
            detr_class = DeformableDETRDINO
        else:
            detr_class = DeformableDETR
        model = detr_class(
        backbone,
        transformer,
        num_queries=self.num_queries,
        num_feature_levels=num_feature_levels,
        aux_loss=deep_supervision,
        with_box_refine=True,
        two_stage=two_stage,
        mixed_selection=cfg.MODEL.DDETRS.MIXED_SELECTION,
        cfg=cfg)

        # Language (text encoder and tokenizer)
        # Here we use BERT as the text encoder in a hard-code way
        self.tokenizer = AutoTokenizer.from_pretrained("checkpoints/bert-base-uncased")
        self.text_encoder = nn.Sequential(OrderedDict([("body", BertEncoder(cfg))]))
        if cfg.MODEL.FREEZE_TEXT_ENCODER:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        
        # backbone for the template branch (for SOT and VOS)
        if cfg.SOT.EXTRA_BACKBONE_FOR_TEMPLATE:
            cfg.defrost()
            cfg.MODEL.CONVNEXT.USE_CHECKPOINT = False
            d2_backbone_ref = MaskedBackbone(cfg, input_shape=ShapeSpec(channels=4))
            ref_backbone = Joiner(d2_backbone_ref, PositionEmbeddingSine(N_steps, normalize=True))
            ref_backbone.num_channels = d2_backbone.num_channels  # only take [c3 c4 c5] from resnet and gengrate c6 later
            ref_backbone.strides = d2_backbone.feature_strides
            model.ref_backbone = ref_backbone
        # DETR + Segmentation (CondInst)
        if cfg.MODEL.DDETRS.USE_DINO:
            model_class = DDETRSegmUniVIDDN
        else:
            model_class = DDETRSegmUniVID
        self.detr = model_class(model, rel_coord=self.use_rel_coord, ota=self.ota, 
        new_mask_head=self.new_mask_head, use_raft=self.use_raft, mask_out_stride=self.mask_stride, 
        template_sz=cfg.SOT.TEMPLATE_SZ, 
        extra_backbone_for_template=cfg.SOT.EXTRA_BACKBONE_FOR_TEMPLATE, search_area_factor=cfg.SOT.SEARCH_AREA_FACTOR,
        ref_feat_sz=cfg.SOT.REF_FEAT_SZ, sot_feat_fusion=cfg.SOT.FEAT_FUSE,
        use_iou_branch=cfg.MODEL.USE_IOU_BRANCH, decouple_tgt=cfg.MODEL.DECOUPLE_TGT, cfg=cfg)

        self.detr.to(self.device)

        # building criterion
        matcher = HungarianMatcherVL(
            cost_class=set_cost_class,
            cost_bbox=set_cost_bbox,
            cost_giou=set_cost_giou)

        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou":giou_weight, \
            "loss_mask": mask_weight, "loss_dice": dice_weight, "loss_reid": reid_weight, "loss_reid_aux":reid_weight*1.5}

        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        if cfg.MODEL.DDETRS.USE_DINO:
            weight_dict_dn = {"loss_ce_dn": class_weight, "loss_bbox_dn": l1_weight, "loss_giou_dn": giou_weight}
            aux_weight_dict_dn = {}
            for i in range(dec_layers - 1):
                aux_weight_dict_dn.update({k + f"_{i}": v for k, v in weight_dict_dn.items()})
            weight_dict_dn.update(aux_weight_dict_dn)
            weight_dict.update(weight_dict_dn)

        losses = ['labelsVL', 'boxes', 'masks', 'reid']
        if cfg.MODEL.DDETRS.USE_DINO:
            criterion_class = DINOCriterion
        else:
            criterion_class = SetCriterion
        self.criterion = criterion_class(matcher, weight_dict, losses, focal_alpha=focal_alpha, ota=self.ota, 
        still_cls_for_encoder=cfg.MODEL.STILL_CLS_FOR_ENCODER, cfg=cfg)
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
        self.use_lsj = cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj"
        # tracker params
        self.init_score_thr = cfg.TRACK.INIT_SCORE_THR # score threshold to start a new track
        self.obj_score_thr = cfg.TRACK.OBJ_SCORE_THR # score threshold to continue a track
        # SOT inference params
        self.online_update = cfg.SOT.ONLINE_UPDATE
        self.update_interval = cfg.SOT.UPDATE_INTERVAL
        self.update_thr = cfg.SOT.UPDATE_THR
        # for SOT and VOS
        self.extra_backbone_for_template = cfg.SOT.EXTRA_BACKBONE_FOR_TEMPLATE
        self.inference_on_3f = cfg.SOT.INFERENCE_ON_3F
        self.inst_thr_vos = cfg.SOT.INST_THR_VOS


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """

        # images = self.preprocess_image(batched_inputs)
        # output = self.detr(images)
        task_list = [x["task"] for x in batched_inputs]
        assert len(set(task_list)) == 1
        task = task_list[0]

        # NOTE: inference only
        if task in ['vis', 'rvos']:
            captions = []
            for video in batched_inputs:
                for cap in video["expressions"]:
                    captions.append(cap)
            assert len(set(captions)) == 1
            positive_map_label_to_token = batched_inputs[0]["positive_map_label_to_token"] 
            num_classes = len(positive_map_label_to_token)
            language_dict_features = self.forward_text(captions[0:1], device="cuda")
            self.tracker = IDOL_Tracker(
                init_score_thr= 0.2,
                obj_score_thr=0.1,
                nms_thr_pre=0.5, 
                nms_thr_post=0.05,
                addnew_score_thr = 0.2,
                memo_tracklet_frames = 10,
                memo_momentum = 0.8,
                long_match = self.inference_tw,
                frame_weight = (self.inference_tw|self.inference_fw),
                temporal_weight = self.inference_tw,
                memory_len = self.memory_len
            )

            # bs = 1 during inference
            height = batched_inputs[0]['height']
            width = batched_inputs[0]['width']
            video_len = len(batched_inputs[0]["image"])
            video_dict = {}
            # process for each frame
            for frame_idx in range(video_len):
                print(f"processing {frame_idx+1} / {video_len} frame...")
                clip_inputs = [{'image':batched_inputs[0]['image'][frame_idx:frame_idx+1]}]
                images = self.preprocess_video(clip_inputs)
                language_dict_features_cur = copy.deepcopy(language_dict_features) # Important
                output, _ = self.detr.coco_inference(images, None, None, language_dict_features=language_dict_features_cur, task='detection')
                output_h, output_w = self.inference_vis(output, positive_map_label_to_token, num_classes, video_dict, frame_idx, (height, width), images.image_sizes[0])
            
            # post process
            video_output = self.post_process_vis(video_dict, video_len, (height, width), images.image_sizes[0], output_h, output_w)
            return video_output

        elif task in ['vos', 'sot']:
            positive_map_label_to_token = {1: [0]}
            num_classes = len(positive_map_label_to_token)

            # bs = 1 during inference
            height = batched_inputs[0]['height']
            width = batched_inputs[0]['width']
            video_len = len(batched_inputs[0]["image"])
            mask_anno = batched_inputs[0]['mask_anno']
            final_masks = []
            for frame_idx in range(video_len):
                print(f"processing {frame_idx+1} / {video_len} frame...")
                clip_inputs = [{'image':batched_inputs[0]['image'][frame_idx:frame_idx+1]}]
                images = self.preprocess_video(clip_inputs)
                # first frame
                if frame_idx == 0:
                    # mask_anno
                    x1_c, y1_c, w_c, h_c = bounding_box(mask_anno) # current bounding box
                    cur_ref_bboxes = torch.tensor([x1_c, y1_c, x1_c+w_c, y1_c+h_c]).view(1, 4)
                    cur_ref_bboxes = [cur_ref_bboxes.to(self.device)] # List (1, 4)
                    size_divisibility = getattr(self.detr.detr.backbone[0].backbone, "size_divisibility", 32)
                    if size_divisibility != 0:
                        mask_h, mask_w = mask_anno.shape
                        mask_h_new = (mask_h + (size_divisibility - 1)) // size_divisibility * size_divisibility
                        mask_w_new = (mask_w + (size_divisibility - 1)) // size_divisibility * size_divisibility
                        mask_anno_new = np.zeros((mask_h_new, mask_w_new), dtype=np.uint8)
                        mask_anno_new[:mask_h, :mask_w] = mask_anno
                        cur_ref_masks = [torch.from_numpy(mask_anno_new[None]).to(self.device)]
                    else:
                        cur_ref_masks = [torch.from_numpy(mask_anno[None]).to(self.device)]
                    self.language_dict_features, template = self.detr.coco_inference_ref_vos(images, cur_ref_bboxes, cur_ref_masks)
                    self.language_dict_features_prev = copy.deepcopy(self.language_dict_features)
                    continue
                # subsequent frames
                if self.online_update:
                    language_dict_features1 = copy.deepcopy(self.language_dict_features)      # Important
                    language_dict_features2 = copy.deepcopy(self.language_dict_features_prev) # Important
                    language_dict_features_cur = {}
                    language_dict_features_cur["hidden"] = torch.cat([language_dict_features1["hidden"], language_dict_features2["hidden"]], dim=1)
                    language_dict_features_cur["masks"] = torch.cat([language_dict_features1["masks"], language_dict_features2["masks"]], dim=1)
                else:
                    language_dict_features_cur = copy.deepcopy(self.language_dict_features)   # Important
                output, _ = self.detr.coco_inference(images, None, None, language_dict_features=language_dict_features_cur, task='sot')
                box_cls = output["pred_logits"]
                box_pred = output["pred_boxes"]
                mask_pred = output["pred_masks"] if self.mask_on else [None] * len(batched_inputs)
                if self.detr.use_iou_branch:
                    iou_pred = output["pred_boxious"]
                else:
                    iou_pred = [None]
                results = self.inference(box_cls, box_pred, mask_pred, images.image_sizes, positive_map_label_to_token, num_classes, task='sot', iou_pred=iou_pred)
                for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                    scale_x, scale_y = (
                        width / results_per_image.image_size[1],
                        height / results_per_image.image_size[0],
                    )
                    results_per_image.pred_boxes.scale(scale_x, scale_y)
                    results_per_image.pred_boxes.clip((height, width))
                    x1, y1, x2, y2 = results_per_image.pred_boxes.tensor.tolist()[0]
                    # mask
                    if results_per_image.scores.item() < self.inst_thr_vos:
                        final_mask = np.zeros((height, width)).astype(np.uint8)
                    else:
                        final_mask = F.interpolate(results_per_image.pred_masks[:,:,:image_size[0],:image_size[1]].float(), size=(height, width), mode="bilinear", align_corners=False)
                        final_mask = final_mask[0, 0].cpu().numpy().astype(np.uint8) # (H, W) original size
                final_masks.append(final_mask)  # topk = 1 for sot
                if self.online_update and (frame_idx % self.update_interval == 0) and (results[0].scores > self.update_thr):
                    # update the template
                    bboxes_unorm = torch.tensor([[x1, y1, x2, y2]]) / torch.tensor([scale_x, scale_y, scale_x, scale_y])
                    # pad ref mask to size_divisibility
                    cur_ref_mask = results_per_image.pred_masks.float()[0]  # [1, h, w]
                    cur_h, cur_w = cur_ref_mask.shape[-2:]
                    pad_h = (cur_h + (size_divisibility - 1)) // size_divisibility * size_divisibility
                    pad_w = (cur_w + (size_divisibility - 1)) // size_divisibility * size_divisibility
                    pad_right = pad_w - cur_w
                    pad_bottom = pad_h - cur_h
                    cur_ref_mask = F.pad(cur_ref_mask, (0, pad_right, 0, pad_bottom), mode='constant', value=0)
                    self.language_dict_features_prev, new_template = self.detr.coco_inference_ref_vos(images, [bboxes_unorm.to(self.device)], [cur_ref_mask])
            return final_masks  # list[np.array], each array of original [H, W], type uint8
                




    def preprocess_clip_image(self, batched_inputs, clip_idx=None):
        """
        Normalize, pad and batch the input images.
        """
        if clip_idx is None:
            images = []
            for video in batched_inputs:
                for frame in video["image"]:
                    images.append(self.normalizer(frame.to(self.device)))
            images = ImageList.from_tensors(images)
        else:
            images = []
            for video in batched_inputs:
                for idx in clip_idx:
                    images.append(self.normalizer(video["image"][idx].to(self.device)))
            images = ImageList.from_tensors(images)
        return images



    def debug_template(self, samples, frame_idx):
        import numpy as np
        import cv2
        mean = np.array([123.675, 116.280, 103.530])
        std = np.array([58.395, 57.120, 57.375])
        assert len(samples.tensors) == 1
        i = 0
        image = samples.tensors[i].permute((1, 2, 0)).cpu().numpy() * std + mean # (H, W, 3)
        input_mask = samples.mask[i].float().cpu().numpy() * 255 # (H, W)
        image = np.ascontiguousarray(image[:, :, ::-1]).clip(0, 255)
        cv2.imwrite("template_%05d_img.jpg"%(frame_idx), image)
        cv2.imwrite("template_%05d_mask.jpg"%(frame_idx), input_mask)

    def debug_template_4c(self, samples, vid_name, obj_id, frame_idx):
        import numpy as np
        import cv2
        mean = np.array([123.675, 116.280, 103.530])
        std = np.array([58.395, 57.120, 57.375])
        assert len(samples.tensors) == 1
        i = 0
        image_mask = samples.tensors[i].permute((1, 2, 0)).cpu().numpy() # [h, w, 4], template_sz
        image = image_mask[:, :, :3]
        gt_mask = image_mask[:, :, -1]
        image = image * std + mean # (H, W, 3)
        input_mask = samples.mask[i].float().cpu().numpy() * 255 # (H, W)
        image = np.ascontiguousarray(image[:, :, ::-1]).clip(0, 255)
        image[:, :, -1] = np.clip(image[:, :, -1] + 100 * gt_mask, 0, 255)
        cv2.imwrite("%s_frame_%05d_obj_%s_img.jpg"%(vid_name, frame_idx, obj_id), image)
        cv2.imwrite("%s_frame_%05d_obj_%s_mask.jpg"%(vid_name, frame_idx, obj_id), input_mask)

    def prepare_targets(self, targets):
        new_targets = []
        # padding gt_masks to max size over a batch (This is important for training with img pairs)
        if hasattr(targets[0], "gt_masks"):
            # mask size: (n_inst, hm, wm)
            gt_masks_list = [x.gt_masks if self.use_lsj else x.gt_masks.tensor for x in targets]
            max_size = _max_by_axis([list(m.shape[1:]) for m in gt_masks_list])
            stride = 32 # size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size[-2] = (max_size[-2] + (stride - 1)) // stride * stride
            max_size[-1] = (max_size[-1] + (stride - 1)) // stride * stride
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            inst_ids = targets_per_image.gt_ids
            valid_id = inst_ids!=-1  # if a object is disappeared，its gt_ids is -1
            # for language-guided detection, classification loss is computed based on the positive map
            positive_map = targets_per_image.positive_map # (N, 256) or (1, 1). N is number of objects per image
            if self.use_amp:
                gt_boxes = gt_boxes.half()
                image_size_xyxy = image_size_xyxy.half()
            if hasattr(targets_per_image, "gt_masks"):
                if self.use_lsj:
                    gt_masks = targets_per_image.gt_masks
                else:
                    gt_masks = targets_per_image.gt_masks.tensor
                if self.use_amp:
                    gt_masks = gt_masks.half()
                # add padding to masks
                n_inst, hm, wm = gt_masks.size()
                gt_masks_pad = torch.zeros((n_inst, max_size[0], max_size[1]), device=gt_masks.device, dtype=gt_masks.dtype)
                gt_masks_pad[:, :hm, :wm].copy_(gt_masks)
                new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'masks': gt_masks_pad, "image_size": image_size_xyxy, 
                "positive_map": positive_map, 'inst_id':inst_ids, "valid": valid_id, "bboxes_unorm": targets_per_image.gt_boxes.tensor})
            else:
                new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "image_size": image_size_xyxy, 
                "positive_map": positive_map, 'inst_id':inst_ids, "valid": valid_id, "bboxes_unorm": targets_per_image.gt_boxes.tensor})
        bz = len(new_targets) // 2
        key_ids = list(range(0, len(new_targets), 2))
        ref_ids = list(range(1, len(new_targets), 2))
        det_targets = [new_targets[_i] for _i in key_ids] # targets on key frames
        ref_targets = [new_targets[_i] for _i in ref_ids] # targets on ref frames
        for i in range(bz):  # fliter empety object in key frame. Note that det_loss is only computed on the key frame !
            det_target = det_targets[i]
            ref_target = ref_targets[i]
            if False in det_target['valid']:
                valid_i = det_target['valid'].clone()
                for k,v in det_target.items():
                    if k != "image_size":
                        det_target[k] = v[valid_i]
                for k,v in ref_target.items():
                    if k != "image_size":
                        ref_target[k] = v[valid_i]


        return det_targets,ref_targets

    def prepare_targets_test(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)  # normalized in [0, 1], x1y1x2y2
            # for language-guided detection, classification loss is computed based on the positive map
            positive_map = torch.ones((len(targets_per_image), 1), dtype=torch.bool, device=self.device) # (N, 256) or (1, 1). N is number of objects per image
            if self.use_amp:
                gt_boxes = gt_boxes.half()
                image_size_xyxy = image_size_xyxy.half()
            if hasattr(targets_per_image, "gt_masks"):
                # padding gt_masks to max size over a batch (This is important for training with img pairs)
                # mask size: (n_inst, hm, wm)
                gt_masks_list = [targets_per_image.gt_masks if self.use_lsj else targets_per_image.gt_masks.tensor]
                max_size = _max_by_axis([list(m.shape[1:]) for m in gt_masks_list])
                stride = 32 # size_divisibility
                # the last two dims are H,W, both subject to divisibility requirement
                max_size[-2] = (max_size[-2] + (stride - 1)) // stride * stride
                max_size[-1] = (max_size[-1] + (stride - 1)) // stride * stride
                if self.use_lsj:
                    gt_masks = targets_per_image.gt_masks
                else:
                    gt_masks = targets_per_image.gt_masks.tensor
                if self.use_amp:
                    gt_masks = gt_masks.half()
                # add padding to masks
                n_inst, hm, wm = gt_masks.size()
                gt_masks_pad = torch.zeros((n_inst, max_size[0], max_size[1]), device=gt_masks.device, dtype=gt_masks.dtype)
                gt_masks_pad[:, :hm, :wm].copy_(gt_masks)
                new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'masks': gt_masks_pad, "image_size": image_size_xyxy, 
                "positive_map": positive_map, "bboxes_unorm": targets_per_image.gt_boxes.tensor})
            else:
                new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "image_size": image_size_xyxy, 
                "positive_map": positive_map, "bboxes_unorm": targets_per_image.gt_boxes.tensor})

        return new_targets

    def inference(self, box_cls, box_pred, mask_pred, image_sizes, positive_map_label_to_token, num_classes, score_thres=0.0, task=None, binary_mask=True, iou_pred=None):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        if task == "detection":
            max_num_inst = 100
        elif task == "grounding" or task == "sot":
            max_num_inst = 1
        else:
            raise ValueError("task must be detection or grounding")
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)
        for (logits_per_image, box_pred_per_image, mask_pred_per_image, image_size, iou_per_image) in zip(
            box_cls, box_pred, mask_pred, image_sizes, iou_pred):

            if self.ota:
                # NMS
                logits_per_image = convert_grounding_to_od_logits(logits_per_image.unsqueeze(0), num_classes, positive_map_label_to_token)
                logits_per_image = logits_per_image[0] # (num_query, C)
                prob = logits_per_image.sigmoid()
                # cls x iou
                if iou_per_image is not None:
                    prob = torch.sqrt(prob * iou_per_image.sigmoid()) # (num_query, C)
                # filter low-score predictions
                if score_thres > 0.0:
                    valid_mask = (prob > score_thres)
                    num_valid = torch.sum(valid_mask).item()
                    num_inst = min(num_valid, max_num_inst)
                    prob[~valid_mask] = -1.0 # set to invalid status
                else:
                    num_inst = max_num_inst
                nms_scores,idxs = torch.max(prob,1)
                boxes_before_nms = box_cxcywh_to_xyxy(box_pred_per_image)
                keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,0.7)  
                prob = prob[keep_indices]
                box_pred_per_image = box_pred_per_image[keep_indices]
                if mask_pred_per_image is not None:
                    mask_pred_per_image = mask_pred_per_image[keep_indices]

                topk_values, topk_indexes = torch.topk(prob.view(-1), num_inst, dim=0)
                scores = topk_values
                topk_boxes = torch.div(topk_indexes, logits_per_image.shape[1], rounding_mode='floor')
                # topk_boxes = topk_indexes // logits_per_image.shape[1]
                labels = topk_indexes % logits_per_image.shape[1]
                scores_per_image = scores
                labels_per_image = labels

                box_pred_per_image = box_pred_per_image[topk_boxes]
                if mask_pred_per_image is not None:
                    mask_pred_per_image = mask_pred_per_image[topk_boxes]
            else:
                logits_per_image = convert_grounding_to_od_logits(logits_per_image.unsqueeze(0), num_classes, positive_map_label_to_token)
                logits_per_image = logits_per_image[0] # (num_query, C)
                prob = logits_per_image.sigmoid()
                # cls x iou
                if iou_per_image is not None:
                    prob = torch.sqrt(prob * iou_per_image.sigmoid()) # (num_query, C)
                # filter low-score predictions
                if score_thres > 0.0:
                    valid_mask = (prob > score_thres)
                    num_valid = torch.sum(valid_mask).item()
                    num_inst = min(num_valid, max_num_inst)
                    prob[~valid_mask] = -1.0 # set to invalid status
                else:
                    num_inst = max_num_inst
                topk_values, topk_indexes = torch.topk(prob.view(-1), num_inst, dim=0)
                scores = topk_values
                topk_boxes = torch.div(topk_indexes, logits_per_image.shape[1], rounding_mode='floor')
                # topk_boxes = topk_indexes // logits_per_image.shape[1]
                labels = topk_indexes % logits_per_image.shape[1]

                scores_per_image = scores
                labels_per_image = labels

                box_pred_per_image = box_pred_per_image[topk_boxes]
                if mask_pred_per_image is not None:
                    mask_pred_per_image = mask_pred_per_image[topk_boxes]
            
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                N, C, H, W = mask_pred_per_image.shape
                mask = F.interpolate(mask_pred_per_image, size=(H*self.mask_stride, W*self.mask_stride), mode='bilinear', align_corners=False)
                mask = mask.sigmoid()
                if binary_mask:
                    mask = mask > self.mask_thres
                mask = mask[:,:,:image_size[0],:image_size[1]]
                result.pred_masks = mask
                
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def inference_mot(self, outputs, positive_map_label_to_token, num_classes, results, i_frame, image_sizes, ori_size, mots=False):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes (without padding)
            ori_size: the original image size in the dataset

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        # results = []
        vido_logits = outputs['pred_logits'] # (Nf, 300, C) Nf=1 for online fashion
        video_output_masks = outputs['pred_masks'] # ([Nf, 300, 1, H//4, W//4)
        output_h, output_w = video_output_masks.shape[-2:]
        video_output_boxes = outputs['pred_boxes'] # (Nf, 300, 4)
        video_output_embeds = outputs['pred_inst_embed'] # (Nf, 300, d)
        if self.detr.use_iou_branch:
            output_ious = outputs["pred_boxious"] # (Nf, 300, 1)
        else:
            output_ious = [None]
        assert len(vido_logits) == 1
        for _, (logits, output_mask, output_boxes, output_embed, image_size, output_iou) in enumerate(zip(
            vido_logits, video_output_masks, video_output_boxes, video_output_embeds, image_sizes, output_ious
         )):
            logits = convert_grounding_to_od_logits(logits.unsqueeze(0), num_classes, positive_map_label_to_token)
            logits = logits[0]
            scores = logits.sigmoid()  #[300,42]
            if output_iou is not None:
                scores = torch.sqrt(scores * output_iou.sigmoid())
            max_score, output_labels = torch.max(scores,1)
            indices = torch.nonzero(max_score>self.inference_select_thres, as_tuple=False).squeeze(1)
            if len(indices) == 0:
                topkv, indices_top1 = torch.topk(scores.cpu().detach().max(1)[0],k=1)
                indices_top1 = indices_top1[torch.argmax(topkv)]
                indices = [indices_top1.tolist()]
            else:
                nms_scores,idxs = torch.max(scores[indices],1)
                boxes_before_nms = box_cxcywh_to_xyxy(output_boxes[indices])
                keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,0.7) #.tolist()
                indices = indices[keep_indices]
            box_score = torch.max(scores[indices],1)[0]
            # [0, 1] -> real coordinates
            output_boxes[:, 0::2] *= ori_size[1]
            output_boxes[:, 1::2] *= ori_size[0]
            det_bboxes = torch.cat([box_cxcywh_to_xyxy(output_boxes[indices]),box_score.unsqueeze(1)],dim=1)
            det_labels = torch.argmax(scores[indices],dim=1)
            track_feats = output_embed[indices]
            det_masks = output_mask[indices]
            if len(indices) > 0:
                track_bboxes, track_labels, ids, indices = self.tracker.match(
                bboxes=det_bboxes,
                labels=det_labels,
                track_feats=track_feats,
                frame_id=i_frame,
                indices=indices)
                valid_mask = ids>-1
                indices = torch.tensor(indices)[valid_mask].tolist()
                ids = ids[valid_mask]
                track_bboxes = track_bboxes[valid_mask]
                track_labels = track_labels[valid_mask]
                track_masks = output_mask[indices] # (N_obj, 1, H, W)
                track_masks = F.interpolate(track_masks,  size=(output_h*4, output_w*4) ,mode="bilinear", align_corners=False).sigmoid().to(self.merge_device)
                track_masks = track_masks[:, :, :image_size[0],:image_size[1]] # crop the padding area
                track_masks = F.interpolate(track_masks, size=(ori_size[0], ori_size[1]), mode='nearest') # (1, 1, H, W)
                track_masks = (track_masks[:, 0] > 0.5).cpu().numpy() # (N, H, W)
                bbox_result = bbox2result(det_bboxes, det_labels, num_classes)
                if mots:
                    track_result = segtrack2result(track_bboxes, track_labels, track_masks, ids)
                    track_result = encode_track_results(track_result)
                else:
                    track_result = track2result(track_bboxes, track_labels, ids, num_classes)
            else:
                if mots:
                    track_result = defaultdict(list)
                    bbox_result = bbox2result(np.zeros((0, 5)), None, num_classes)
                    # segm_result = [[] for _ in range(num_classes)]
                    # result = {"track_result": track_result, "bbox_result": bbox_result, "segm_result": segm_result}
                else:
                    track_masks = np.zeros((0, ori_size[0], ori_size[1]))
                    bbox_result = bbox2result(np.zeros((0, 5)), None, num_classes)
                    track_result = [np.zeros((0, 6), dtype=np.float32) for i in range(num_classes)]
            # result = dict(bbox_results=bbox_result, track_results=track_result, mask_results=track_masks)
            if mots:
                results["bbox_result"].append(bbox_result)
                results["track_result"].append(track_result)
            else:
                results["bbox_results"].append(bbox_result)
                results["track_results"].append(track_result)
        del outputs




    def match_from_embds(self, tgt_embds, cur_embds):

        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0,1))

        cost_embd = 1 - cos_sim

        C = 1.0 * cost_embd
        C = C.cpu()

        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target

        return indices

    def inference_vis(self, outputs, positive_map_label_to_token, num_classes, video_dict, i_frame, ori_size, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        # results = []
        vido_logits = outputs['pred_logits'] # (Nf, 300, C) Nf=1 for online fashion
        video_output_masks = outputs['pred_masks'] # ([Nf, 300, 1, H//4, W//4)
        output_h, output_w = video_output_masks.shape[-2:]
        video_output_boxes = outputs['pred_boxes'] # (Nf, 300, 4)
        video_output_embeds = outputs['pred_inst_embed'] # (Nf, 300, d)
        if self.detr.use_iou_branch:
            output_ious = outputs["pred_boxious"] # (Nf, 300, 1)
        else:
            output_ious = [None]
        for _, (logits, output_mask, output_boxes, output_embed, output_iou) in enumerate(zip(
            vido_logits, video_output_masks, video_output_boxes, video_output_embeds, output_ious
         )):
            logits = convert_grounding_to_od_logits(logits.unsqueeze(0), num_classes, positive_map_label_to_token)
            logits = logits[0]
            if output_iou is not None:
                scores = torch.sqrt(logits.sigmoid() * output_iou.sigmoid()).cpu().detach()  #[300,42]
                max_score, _ = torch.max(torch.sqrt(logits.sigmoid() * output_iou.sigmoid()),1)
            else:
                scores = logits.sigmoid().cpu().detach()  #[300,42]
                max_score, _ = torch.max(logits.sigmoid(),1)
            indices = torch.nonzero(max_score>self.inference_select_thres, as_tuple=False).squeeze(1)
            if len(indices) == 0:
                topkv, indices_top1 = torch.topk(scores.max(1)[0],k=1)
                indices_top1 = indices_top1[torch.argmax(topkv)]
                indices = [indices_top1.tolist()]
            else:
                if output_iou is not None:
                    nms_scores,idxs = torch.max(torch.sqrt(logits.sigmoid() * output_iou.sigmoid())[indices],1)
                else:
                    nms_scores,idxs = torch.max(logits.sigmoid()[indices],1)
                boxes_before_nms = box_cxcywh_to_xyxy(output_boxes[indices])
                keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,0.9)#.tolist()
                indices = indices[keep_indices]
            if output_iou is not None:
                box_score = torch.max(torch.sqrt(logits.sigmoid() * output_iou.sigmoid())[indices],1)[0]
                det_labels = torch.argmax(torch.sqrt(logits.sigmoid() * output_iou.sigmoid())[indices],dim=1)
            else:
                box_score = torch.max(logits.sigmoid()[indices],1)[0]
                det_labels = torch.argmax(logits.sigmoid()[indices],dim=1)
            det_bboxes = torch.cat([output_boxes[indices],box_score.unsqueeze(1)],dim=1)
            track_feats = output_embed[indices]
            det_masks = output_mask[indices]
            bboxes, labels, ids, indices = self.tracker.match(
            bboxes=det_bboxes,
            labels=det_labels,
            masks = det_masks,
            track_feats=track_feats,
            frame_id=i_frame,
            indices = indices)
            indices = torch.tensor(indices)[ids>-1].tolist()
            ids = ids[ids > -1]
            ids = ids.tolist()
            for query_i, id in zip(indices,ids):
                mask_i = output_mask[query_i]
                # upsample, resize back to the original image size, transform to rle
                pred_mask_i = F.interpolate(mask_i[:,None,:,:],  size=(output_h*4, output_w*4) ,mode="bilinear", align_corners=False).sigmoid().to(self.merge_device)
                pred_mask_i = pred_mask_i[:, :, :image_sizes[0],:image_sizes[1]] # crop the padding area
                pred_mask_i = F.interpolate(pred_mask_i, size=(ori_size[0], ori_size[1]), mode='nearest') # (1, 1, H, W)
                pred_mask_i = pred_mask_i[0, 0] > 0.5 # (H, W)
                pred_mask_i_rle = mask_util.encode(np.array(pred_mask_i[:, :, None].cpu(), order="F", dtype="uint8"))[0]
                if id in video_dict.keys():
                    video_dict[id]['masks'].append(pred_mask_i_rle)
                    video_dict[id]['boxes'].append(output_boxes[query_i])
                    video_dict[id]['scores'].append(scores[query_i])
                    video_dict[id]['valid'] = video_dict[id]['valid'] + 1
                else:
                    video_dict[id] = {
                        'masks':[None for fi in range(i_frame)], 
                        'boxes':[None for fi in range(i_frame)], 
                        'scores':[None for fi in range(i_frame)], 
                        'valid':0}
                    video_dict[id]['masks'].append(pred_mask_i_rle)
                    video_dict[id]['boxes'].append(output_boxes[query_i])
                    video_dict[id]['scores'].append(scores[query_i])
                    video_dict[id]['valid'] = video_dict[id]['valid'] + 1

            for k,v in video_dict.items():
                if len(v['masks'])<i_frame+1: #padding None for unmatched ID
                    v['masks'].append(None)
                    v['scores'].append(None)
                    v['boxes'].append(None)
            check_len = [len(v['masks']) for k,v in video_dict.items()]
            # print('check_len',check_len)

            #  filtering sequences that are too short in video_dict (noise)，the rule is: if the first two frames are None and valid is less than 3
            if i_frame>8:
                del_list = []
                for k,v in video_dict.items():
                    if v['masks'][-1] is None and  v['masks'][-2] is None and v['valid']<3:
                        del_list.append(k)   
                for del_k in del_list:
                    video_dict.pop(del_k)                      

        del outputs
        torch.cuda.empty_cache()

        return output_h, output_w

    def post_process_vis(self, video_dict, vid_len, ori_size, image_sizes, output_h, output_w):
        logits_list = []
        masks_list = []
        for inst_id, m in  enumerate(video_dict.keys()):
            score_list_ori = video_dict[m]['scores']
            scores_temporal = []
            for k in score_list_ori:
                if k is not None:
                    scores_temporal.append(k)
            logits_i = torch.stack(scores_temporal)
            if self.temporal_score_type == 'mean':
                logits_i = logits_i.mean(0)
            elif self.temporal_score_type == 'max':
                logits_i = logits_i.max(0)[0]
            else:
                print('non valid temporal_score_type')
                import sys;sys.exit(0)
            logits_list.append(logits_i)
            
            # category_id = np.argmax(logits_i.mean(0))
            masks_list_i = []
            for n in range(vid_len):
                mask_i = video_dict[m]['masks'][n] # (1, h//4, w//4) or rle
                if mask_i is None:
                    masks_list_i.append(None)
                else:
                    masks_list_i.append(mask_i)
            masks_list.append(masks_list_i)
        if len(logits_list)>0:
            pred_cls = torch.stack(logits_list)
        else:
            pred_cls = []

        # pred_cls: (num_obj, C)
        # pred_masks: (num_obj, num_frame, H, W)
        if len(pred_cls) > 0:
            if self.is_multi_cls:
                # is_above_thres is a tuple of 1-D tensors, 
                # one for each dimension in input, 
                # each containing the indices (in that dimension) of all non-zero elements of input
                is_above_thres = torch.where(pred_cls > self.apply_cls_thres)
                scores = pred_cls[is_above_thres] # (num_obj, )
                labels = is_above_thres[1] # (num_obj, )
                masks_list_mc = [] # masks_list multi_cls
                for idx in is_above_thres[0]:
                    masks_list_mc.append(masks_list[idx])
                out_masks = masks_list_mc
            else:
                scores, labels = pred_cls.max(-1)
                out_masks = masks_list
            out_scores = scores.tolist()
            out_labels = labels.tolist()
        else:
            out_scores = []
            out_labels = []
            out_masks = []
        video_output = {
            "image_size": ori_size,
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }
        return video_output


    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        if self.use_lsj and self.training:
            image_sizes = [x["instances"].image_size for x in batched_inputs]
            input_masks = [x["padding_mask"].to(self.device) for x in batched_inputs]
            H, W = images[0].size()[-2:]
            images_new = torch.zeros((len(images), 3, H, W), device=self.device)
            for i in range(len(images)):
                h, w = image_sizes[i]
                images_new[i, :, :h, :w] = images[i][:, :h, :w]
            outputs = NestedTensor(images_new, torch.stack(input_masks, dim=0))
            outputs.image_sizes = image_sizes
            return outputs
        else:
            images = ImageList.from_tensors(images)
            return images

    def preprocess_video(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(self.normalizer(frame.to(self.device)))
        images = ImageList.from_tensors(images)
        return images

    def forward_text(self, captions, device):
        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(captions,
                                                        max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, # 256
                                                        padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest", # max_length
                                                        return_special_tokens_mask=True,
                                                        return_tensors='pt',
                                                        truncation=True).to(device)

            tokenizer_input = {"input_ids": tokenized.input_ids,
                            "attention_mask": tokenized.attention_mask}
            language_dict_features = self.text_encoder(tokenizer_input) # dict with keys: ['aggregate', 'embedded', 'masks', 'hidden']
            # language_dict_features["masks"] is equal to tokenizer_input["attention_mask"]
            # aggregate: (bs, 768), embedded: (bs, L, 768), masks: (bs, 768), hidden: (bs, L, 768) L=256 here
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        return language_dict_features


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout):
        super().__init__()
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        x = self.layer_norm(x)
        output = self.dropout(x)
        return output

def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2-x1), int(y2-y1)] # (x1, y1, w, h) 

def convert_grounding_to_od_logits(logits, num_classes, positive_map, score_agg="MEAN"):
    """
    logits: (bs, num_query, max_seq_len)
    num_classes: 80 for COCO
    """
    assert logits.ndim == 3
    assert positive_map is not None
    scores = torch.zeros(logits.shape[0], logits.shape[1], num_classes).to(logits.device)
    # 256 -> 80, average for each class
    # score aggregation method
    if score_agg == "MEAN": # True
        for label_j in positive_map:
            scores[:, :, label_j - 1] = logits[:, :, torch.LongTensor(positive_map[label_j])].mean(-1)
    else:
        raise NotImplementedError
    return scores

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]

def track2result(bboxes, labels, ids, num_classes):
    valid_inds = ids > -1
    bboxes = bboxes[valid_inds]
    labels = labels[valid_inds]
    ids = ids[valid_inds]

    if bboxes.shape[0] == 0:
        return [np.zeros((0, 6), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()
            labels = labels.cpu().numpy()
            ids = ids.cpu().numpy()
        return [
            np.concatenate((ids[labels == i, None], bboxes[labels == i, :]),
                           axis=1) for i in range(num_classes)
        ]

def segtrack2result(bboxes, labels, segms, ids):
    valid_inds = ids > -1
    bboxes = bboxes[valid_inds].cpu().numpy()
    labels = labels[valid_inds].cpu().numpy()
    segms = [segms[i] for i in range(len(segms)) if valid_inds[i] == True]
    ids = ids[valid_inds].cpu().numpy()

    outputs = defaultdict(list)
    for bbox, label, segm, id in zip(bboxes, labels, segms, ids):
        outputs[id] = dict(bbox=bbox, label=label, segm=segm)
    return outputs

def encode_track_results(track_results):
    """Encode bitmap mask to RLE code.

    Args:
        track_results (list | tuple[list]): track results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).

    Returns:
        list | tuple: RLE encoded mask.
    """
    for id, roi in track_results.items():
        roi['segm'] = mask_util.encode(
            np.array(roi['segm'][:, :, np.newaxis], order='F',
                     dtype='uint8'))[0]  # encoded with RLE
    return track_results