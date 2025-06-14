"""
# prompt for GPT to write questions...
You are a helpful prompt generator. I need you to write several human-friendly and clear prompts for specific vision task with both interrogative and declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 prompts.

Task: Multi-Object Tracking and Segmentation
Examples:
1. "Could you track all instances of <class> with boxes and masks for the video?",
2. "Please provide the tracking boxes and masks for the targets belong to <class> for the video.",
3. "Could you help me detect, segment and track all the <class> objects throughout the video?"
4. "I need your assistance in tracking the <class> with boxes and masks.",

Note: <class> are the placeholder for specific object categories. 
      This task needs both the boxes and masks for the targets.


# prompt for GPT to write answer...
You are a helpful answer generator. I need you to write several human-friendly and clear answers for specific vision task with declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 answers.

Task: Multi-Object Tracking and Segmentation
Examples:
1. "This is a <task>multi-object tracking and segmentation (mots)</task> task. Here are the results for <class> in the video."
2. "This is a <task>multi-object tracking and segmentation (mots)</task> task. I provide the tracking boxes and masks for <class> for the video."

Note: Do not change "This is a <task>multi-object tracking and segmentation (mots)</task> task.". <class> are the placeholder for specific object categories.
      Do not contain specific numbers, such as [x] instances, [x] results. 
      Do not say about the confidence scores.
"""

QUESTIONS = [
    "Could you predict bounding boxes and segmentation masks for all <class> instances in the video sequence?",
    "Can you generate real-time predictions of tracking trajectories and masks for <class> objects?",
    "Would you mind inferring the temporal tracking boxes and corresponding masks for each <class> entity?",
    "Is it possible to predict both box coordinates and pixel-level masks for <class> objects across all frames?",
    "Could you help me obtain predictive tracking results—including boxes and masks—for <class> instances in the video?",
    "Can you analyze the video to predict each <class> object's trajectory with associated boxes and masks?",
    "Would you be able to output predicted tracking boxes and masks for <class> targets in real time?",
    "Could you show me the results for <class> objects, overlaying tracking boxes and segmentation masks?",
    "Can you predict sequential masks and box annotations for <class> objects in the clip?",
    "Can you process the video to generate predictive tracking data (boxes + masks) for every <class> instance?",
    "Please use the model to predict tracking boxes and segmentation masks for all <class> objects in the video.",
    "I need predictive results: track each <class> instance and output its box coordinates and mask per frame.",
    "Kindly generate sequential predictions of <class> objects, including their tracking boxes and corresponding masks.",
    "Please help to detect, track, and predict masks for <class> entities for the video.",
    "I request real-time predictive results for <class> objects: tracking boxes and pixel-wise masks in each frame.",
    "Please output model predictions of <class> object trajectories, paired with their segmentation masks.",
    "I need your help to predict comprehensive tracking results for <class> instances, combining boxes and masks.",
    "Kindly use the model to infer both spatial (box/mask) and temporal (tracking) predictions for <class> objects."
    "Please deliver predictive analysis of <class> objects, including frame-by-frame box coordinates and masks.",
    "I require you to generate predictions for <class> tracking, with boxes and masks visualized in the video sequence.",
]

ANSWERS = [
    "This is a <task>multi-object tracking and segmentation (mots)</task> task. Here are the tracking boxes and masks for <class> in the video.",
    "This is a <task>multi-object tracking and segmentation (mots)</task> task. The tracking results, including boxes and masks, for <class> in the video are presented here.",
    "This is a <task>multi-object tracking and segmentation (mots)</task> task. I provide the tracking boxes and masks that outline <class> in the video.",
    "This is a <task>multi-object tracking and segmentation (mots)</task> task. The tracking outcomes, comprising boxes and masks, for <class> in the video are shown here.",
    "This is a <task>multi-object tracking and segmentation (mots)</task> task. Here are the tracking boxes and masks that identify <class> in the video.",
    "This is a <task>multi-object tracking and segmentation (mots)</task> task. I offer the tracking results—including boxes and masks—for <class> within the video.",
    "This is a <task>multi-object tracking and segmentation (mots)</task> task. The tracking boxes and masks for <class> in the video are provided here.",
    "This is a <task>multi-object tracking and segmentation (mots)</task> task. Here are the tracking results, featuring boxes and masks, for <class> in the video.",
    "This is a <task>multi-object tracking and segmentation (mots)</task> task. I provide the tracking boxes and masks that track <class> in the video.",
    "This is a <task>multi-object tracking and segmentation (mots)</task> task. The tracking outcomes, including boxes and masks, for each <class> in the video are presented here.",
    "This is a <task>multi-object tracking and segmentation (mots)</task> task. Here are the tracking boxes and masks generated for <class> in the video.",
    "This is a <task>multi-object tracking and segmentation (mots)</task> task. I offer the tracking results—boxes and masks—for <class> captured in the video.",
    "This is a <task>multi-object tracking and segmentation (mots)</task> task. The tracking boxes and masks that segment <class> in the video are shown here.",
    "This is a <task>multi-object tracking and segmentation (mots)</task> task. Here are the tracking results, including bounding boxes and masks, for <class> in the video.",
    "This is a <task>multi-object tracking and segmentation (mots)</task> task. I provide the tracking boxes and masks that outline each <class> in the video.",
    "This is a <task>multi-object tracking and segmentation (mots)</task> task. The tracking outcomes, comprising boxes and segmentation masks, for <class> are here.",
    "This is a <task>multi-object tracking and segmentation (mots)</task> task. Here are the tracking boxes and masks for <class> across the video frames.",
    "This is a <task>multi-object tracking and segmentation (mots)</task> task. I offer the tracking results—both boxes and masks—for <class> in the video.",
]