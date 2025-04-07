# UNINEXT Evaluation

Evaluation script follows the format as 

```
bash scripts/vllmv2_7b/eval/dist_eval_uninext.sh [vllmv2_dir] [eval_config]
```

在shlab集群上，用srun起命令，bash前加上`s8a`

## 1 OD & IS

```
bash scripts/vllmv2_7b/eval/dist_eval_uninext.sh work_dirs/visionllmv2-7b-pure visionllmv2/datasets/configs/video/vos_val.py 

```

## 2 REC & RES

```
bash scripts/vllmv2_7b/eval/dist_eval_uninext.sh work_dirs/visionllmv2-7b-pure visionllmv2/datasets/configs/video/rec_val.py 
```

## 3 VIS

### YTVIS2019

```
bash scripts/vllmv2_7b/eval/dist_eval_uninext.sh work_dirs/visionllmv2-7b-pure visionllmv2/datasets/configs/video/vis_val.py 

# zip the results
cd outputs/video_joint_vit_huge/inference
zip -q -r VIS19.zip results.json
cd ../../..
```

Then, submit the zip file to the [evaluation server](https://codalab.lisn.upsaclay.fr/competitions/6064#participate-submit_results).


### OVIS

```
bash scripts/vllmv2_7b/eval/dist_eval_uninext.sh work_dirs/visionllmv2-7b-pure visionllmv2/datasets/configs/video/ovis_val.py 

# zip the results
cd outputs/video_joint_vit_huge/inference
zip -q -r OVIS.zip results.json
cd ../../..
```

Then, submit the zip file to the [evaluation server](https://codalab.lisn.upsaclay.fr/competitions/4763#participate-submit_results).


## 4 VOS

```
bash scripts/vllmv2_7b/eval/dist_eval_uninext.sh work_dirs/visionllmv2-7b-pure visionllmv2/datasets/configs/video/vos_val.py 
```

### YTVOS2018

```
cd outputs/video_joint_vit_huge/inference
mv ytbvos18 Annotations
zip -q -r VOS.zip Annotations
cd ../../..
```

Then, submit the zip file to the [evaluation server](https://codalab.lisn.upsaclay.fr/competitions/7685#participate-submit_results).

### DAVIS

```
cd UNINEXT/external/davis2017-evaluation
python3 evaluation_method.py --task semi-supervised --results_path ../../../outputs/video_joint_vit_huge/inference/DAVIS --davis_path ../../../datasets/DAVIS
```

## 5 RVOS

```
bash scripts/vllmv2_7b/eval/dist_eval_uninext.sh work_dirs/visionllmv2-7b-pure visionllmv2/datasets/configs/video/rvos_val.py 
```

### Ref-Youtube-VOS

```
cd outputs/video_joint_vit_huge/inference
mv refytvos Annotations
zip -q -r RVOS.zip Annotations
cd ../../..
```

Then, submit the zip file to the [evaluation server](https://codalab.lisn.upsaclay.fr/competitions/3282#participate-submit_results).

### Ref-DAVIS

```
cd UNINEXTexternal/davis2017-evaluation
python3 evaluation_method.py --task unsupervised --results_path ../../../outputs/video_joint_vit_huge/inference/rvos-refdavis-val-0 --davis_path ../../../datasets/ref-davis/DAVIS
python3 evaluation_method.py --task unsupervised --results_path ../../../outputs/video_joint_vit_huge/inference/rvos-refdavis-val-1 --davis_path ../../../datasets/ref-davis/DAVIS
python3 evaluation_method.py --task unsupervised --results_path ../../../outputs/video_joint_vit_huge/inference/rvos-refdavis-val-2 --davis_path ../../../datasets/ref-davis/DAVIS
python3 evaluation_method.py --task unsupervised --results_path ../../../outputs/video_joint_vit_huge/inference/rvos-refdavis-val-3 --davis_path ../../../datasets/ref-davis/DAVIS

```


## 6 SOT

```
bash scripts/vllmv2_7b/eval/dist_eval_uninext.sh work_dirs/visionllmv2-7b-pure visionllmv2/datasets/configs/video/rvos_val.py 
```

### TrackingNet

```
python3 tools_uninext/transform_trackingnet.py --exp_name video_joint_vit_huge
```

Then, submit the `TrackingNet_submit.zip` to [evaluation server](https://eval.ai/web/challenges/challenge-page/1805/overview).

### LaSOT & LaSOText & TNL2K

Copy the originial results to a new folder.

```
mkdir -p UNINEXT/video_joint_vit_huge
cp outputs/video_joint_vit_huge/inference/LaSOT/* UNINEXT/video_joint_vit_huge
cp outputs/video_joint_vit_huge/inference/LaSOT_extension_subset/* UNINEXT/video_joint_vit_huge
cp outputs/video_joint_vit_huge/inference/TNL-2K/* UNINEXT/video_joint_vit_huge
```

Run the command for evaluation. Specify the dataset `lasot`, `lasotext`, or `tnl2k` for the arg `--dataset_name`.

```
python3 tools_uninext/analysis_results.py --exp_name video_joint_vit_huge --dataset_name [dataset_name]
```


## 7 MOT 

```
bash scripts/vllmv2_7b/eval/dist_eval_uninext.sh work_dirs/visionllmv2-7b-pure visionllmv2/datasets/configs/video/mot_val.py 
```

Then, run the command for evaluation.

```
python3 tools_uninext/eval_bdd.py outputs/video_joint_vit_huge/inference/instances_predictions_init_0.40_obj_0.30.pkl
```


## 8 MOTS

```
bash scripts/vllmv2_7b/eval/dist_eval_uninext.sh work_dirs/visionllmv2-7b-pure visionllmv2/datasets/configs/video/mots_val.py 
```

```
# Install extra packages
pip3 install bdd100k
pip3 install scalabel
# convert to BDD100K format (bitmask)
python3 tools_uninext/to_bdd100k.py --res outputs/video_joint_vit_huge/inference/instances_predictions_init_0.40_obj_0.30.pkl --task seg_track --bdd-dir . --nproc 32
# evaluate
bash tools_uninext/eval_bdd_submit.sh
```