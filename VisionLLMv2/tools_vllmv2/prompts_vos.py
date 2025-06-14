"""
# prompt for GPT to write questions...
You are a helpful prompt generator. I need you to write several human-friendly and clear prompts for specific vision task with both interrogative and declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 prompts.

Task: Video Object Segmentation
Examples:
1. "Could you segment and track the target outlined by the given mask for the video?",
2. "Please provide the tracking masks for the targets with specified mask in the first frame.",
3. "Given the annotated mask for the target object, could you help track the object for the whole video?"
4. "I need your assistance in tracking the target with provided mask.",


# prompt for GPT to write answer...
You are a helpful answer generator. I need you to write several human-friendly and clear answers for specific vision task with declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 answers.

Task: Video Object Segmentation
Examples:
1. "This is a <task>video object segmentation (vos)</task> task. Here are the results for <class> in the video."
2. "This is a <task>video object segmentation (vos)</task> task. I provide the tracking masks for <class> for the video."

Note: Do not change "This is a <task>video object segmentation (vos)</task> task.". 
      Do not contain specific numbers, such as [x] instances, [x] results. 
      Do not say about the confidence scores.
"""

QUESTIONS = [
    "Can you segment and track the moving object in the video using the initial mask provided?",
    "Could you generate pixel-level masks to track the target object throughout the entire video sequence?",
    "Would you mind segmenting the object outlined in the first frame and maintaining its tracking across all video frames?",
    "Do you have the ability to segment and track the object shown in the annotated mask for each frame of the video?",
    "Can you assist in generating a sequence of masks to track the object identified in the initial frame?",
    "Could you perform video object segmentation to track the object highlighted in the first frame's mask?",
    "How would you go about segmenting the object in the video using the provided mask as a starting point?",
    "Are you able to generate tracking masks for the object marked in the given annotation throughout the video?",
    "Can you help me segment and track the object in the video with the mask provided in the first frame?",
    "Please segment and track the target object in the video using the initial mask for reference.",
    "Generate a sequence of segmentation masks to track the object outlined in the first frame's annotation.",
    "Apply video object segmentation to track the object specified by the given mask across all video frames.",
    "I need you to provide pixel-wise masks to track the object highlighted in the initial frame's mask.",
    "Create a tracking mask sequence for the object marked in the annotation to follow it through the video.",
    "Segment and track the object in the video starting from the mask provided in the first frame.",
    "Generate masks to track the moving object in the video, based on the initial segmentation mask given.",
    "I request you to apply video object segmentation to track the target with the mask provided here.",
    "Please ensure the object in the given mask is segmented and tracked throughout the entire video sequence.",
]

ANSWERS = [
    "This is a <task>video object segmentation (vos)</task> task. The segmentation masks in the video are provided here.",
    "This is a <task>video object segmentation (vos)</task> task. Here are the detailed object masks throughout the video.",
    "This is a <task>video object segmentation (vos)</task> task. The tracking and segmentation results in the video are presented.",
    "This is a <task>video object segmentation (vos)</task> task. I offer the pixel-wise masks in each frame of the video.",
    "This is a <task>video object segmentation (vos)</task> task. The generated masks across the video sequence are here.",
    "This is a <task>video object segmentation (vos)</task> task. Here are the object segmentation outputs in the video.",
    "This is a <task>video object segmentation (vos)</task> task. The segmentation results tracking objects in the video are provided.",
    "This is a <task>video object segmentation (vos)</task> task. I have the masks that segment objects in each video frame.",
    "This is a <task>video object segmentation (vos)</task> task. The detailed segmentation masks in the video are here.",
    "This is a <task>video object segmentation (vos)</task> task. Here are the per-frame masks from the video segmentation.",
    "This is a <task>video object segmentation (vos)</task> task. The object segmentation masks tracking objects in the video are presented.",
    "This is a <task>video object segmentation (vos)</task> task. I provide the masks that segment objects throughout the video sequence.",
    "This is a <task>video object segmentation (vos)</task> task. The generated segmentation results in the video are here.",
    "This is a <task>video object segmentation (vos)</task> task. Here are the pixel-precise masks in the video frames.",
    "This is a <task>video object segmentation (vos)</task> task. The segmentation masks following objects in the video are provided.",
    "This is a <task>video object segmentation (vos)</task> task. I have the detailed masks from the video segmentation process.",
    "This is a <task>video object segmentation (vos)</task> task. The per-frame object segmentation in the video is here.",
    "This is a <task>video object segmentation (vos)</task> task. Here are the segmentation outputs that track objects in the video.",
    "This is a <task>video object segmentation (vos)</task> task. The masks segmenting objects across the entire video are presented.",
    "This is a <task>video object segmentation (vos)</task> task. I provide the tracking and segmentation masks in the video.",

]