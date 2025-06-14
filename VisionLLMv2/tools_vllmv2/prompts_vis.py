"""
# prompt for GPT to write questions...
You are a helpful prompt generator. I need you to write several human-friendly and clear prompts for specific vision task with both interrogative and declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 prompts.

Task: Video Instance Segmentation
Examples:
1. "Could you segment and track all instances of <class> for the video?",
2. "Please provide the tracking masks for the targets belong to <class> for the video.",
3. "Could you help me track all the <class> objects with segmentation masks throughout the video?"
4. "I need your assistance in tracking the <class> with masks.",

Note: <class> are the placeholder for specific object categories. 


# prompt for GPT to write answer...
You are a helpful answer generator. I need you to write several human-friendly and clear answers for specific vision task with declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 answers.

Task: Video Instance Segmentation
Examples:
1. "This is a <task>video instance segmentation (vis)</task> task. Here are the results for <class> in the video."
2. "This is a <task>video instance segmentation (vis)</task> task. I provide the tracking masks for <class> for the video."

Note: Do not change "This is a <task>video instance segmentation (vis)</task> task.". <class> are the placeholder for specific object categories.
      Do not contain specific numbers, such as [x] instances, [x] results. 
      Do not say about the confidence scores.
"""

QUESTIONS = [
    "Could you generate instance segmentation masks for all <class> objects in the video and track their movement?",
    "Can you segment each individual <class> instance frame-by-frame and provide tracking trajectories?",
    "Would you mind segmenting and tracking every <class> object throughout the entire video sequence?",
    "Could you help me obtain detailed instance masks for all <class> entities with tracking information?",
    "Can you perform video instance segmentation on <class> objects and output their masked trajectories?",
    "Would you be able to segment each <class> instance and track it across the video frames?",
    "Could you provide segmented masks for every <class> object along with their tracking paths?",
    "Can you generate per-frame instance segmentations for <class> objects in the video?",
    "Would you mind tracking and segmenting all <class> instances from the start to the end of the video?",
    "Could you assist in creating masked tracks for each <class> object in the video?",
    "Please segment and track all instances of <class> in the video with detailed masks.",
    "Provide per-frame instance segmentation masks for every <class> object in the video.",
    "Generate tracking results for <class> objects along with their segmentation masks.",
    "Perform video instance segmentation on <class> entities and include tracking information.",
    "Create masked trajectories for each <class> instance throughout the video.",
    "Segment each <class> object in the video and track its movement frame-by-frame.",
    "Output instance segmentation masks for all <class> objects with tracking data.",
    "Please track and segment <class> instances in the video, providing masked results.",
    "Generate detailed instance segmentations for <class> objects across all video frames.",
    "Provide tracking masks for every <class> instance in the video sequence.",
    "Perform instance segmentation and tracking on <class> objects in the video.",
    "Create per-instance masks for <class> entities with their tracking paths in the video.",
]

ANSWERS = [
    "This is a <task>video instance segmentation (vis)</task> task. Here are the segmented and tracked instances of <class> in the video.",
    "This is a <task>video instance segmentation (vis)</task> task. I have provided the tracking masks for <class> throughout the video.",
    "This is a <task>video instance segmentation (vis)</task> task. The results include per-frame instance masks for <class> objects.",
    "This is a <task>video instance segmentation (vis)</task> task. Here are the masked trajectories for each <class> instance in the video.",
    "This is a <task>video instance segmentation (vis)</task> task. The output contains segmented masks and tracking paths for <class>.",
    "This is a <task>video instance segmentation (vis)</task> task. I've generated instance segmentations for <class> across all video frames.",
    "This is a <task>video instance segmentation (vis)</task> task. The results feature tracked <class> objects with detailed masks.",
    "This is a <task>video instance segmentation (vis)</task> task. Here are the per-instance masks for <class> in the video sequence.",
    "This is a <task>video instance segmentation (vis)</task> task. The output includes segmenting and tracking every <class> object.",
    "This is a <task>video instance segmentation (vis)</task> task. I provide the masked tracking results for <class> in the video.",
    "This is a <task>video instance segmentation (vis)</task> task. The output presents segmented <class> instances with tracking data.",
    "This is a <task>video instance segmentation (vis)</task> task. Here are the frame-by-frame masks for <class> objects.",
    "This is a <task>video instance segmentation (vis)</task> task. The results include tracked <class> instances with segmentation masks.",
    "This is a <task>video instance segmentation (vis)</task> task. I've created masked trajectories for each <class> in the video.",
    "This is a <task>video instance segmentation (vis)</task> task. The work provides per-instance segmentations for <class> across the video.",
    "This is a <task>video instance segmentation (vis)</task> task. Here are the tracking results with masks for <class> objects.",
    "This is a <task>video instance segmentation (vis)</task> task. The output includes segmented and tracked <class> instances.",
    "This is a <task>video instance segmentation (vis)</task> task. I have generated tracking masks for <class> throughout the video frames.",
    "This is a <task>video instance segmentation (vis)</task> task. The results feature detailed masks for each <class> instance tracked in the video.",
    "This is a <task>video instance segmentation (vis)</task> task. Here are the per-frame instance segmentations for <class> with tracking information.",
    "This is a <task>video instance segmentation (vis)</task> task. The work provides masked tracks for <class> objects in the entire video.",
    "This is a <task>video instance segmentation (vis)</task> task. I've output the segmented and tracked <class> instances with their masks.",
]