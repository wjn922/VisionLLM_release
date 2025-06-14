"""
# prompt for GPT to write questions...
You are a helpful prompt generator. I need you to write several human-friendly and clear prompts for specific vision task with both interrogative and declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 prompts.

Task: Multi-Object Tracking
Examples:
1. "Could you detect and track all instances of <class> for the video?",
2. "Please provide the tracking boxes for the targets belong to <class> for the video.",
3. "Could you help me track all the <class> objects with bounding masks throughout the video?"
4. "I need your assistance in tracking the <class> with boxes.",

Note: <class> are the placeholder for specific object categories. 


# prompt for GPT to write answer...
You are a helpful answer generator. I need you to write several human-friendly and clear answers for specific vision task with declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 answers.

Task: Multi-Object Tracking
Examples:
1. "This is a <task>multi-object tracking (mot)</task> task. Here are the results for <class> in the video."
2. "This is a <task>multi-object tracking (mot)</task> task. I provide the tracking boxes for <class> for the video."

Note: Do not change "This is a <task>multi-object tracking (mot)</task> task.". <class> are the placeholder for specific object categories.
      Do not contain specific numbers, such as [x] instances, [x] results. 
      Do not say about the confidence scores.
"""

QUESTIONS = [
    "Can you use bounding boxes to track all instances of <class> in the video?",
    "Could you display the bounding boxes for each tracked <class> object across all frames?",
    "Will you track <class> objects simultaneously using bounding boxes?",
    "Can you ensure accurate bounding box tracking for <class> objects for the video?",
    "Could you generate tracking data with bounding boxes for every <class> object in the video sequence?",
    "Do you support tracking <class> objects with distinct bounding boxes for the video?",
    "Could you help me detect and track <class> objects in the provided scenes using bounding boxes?",
    "Will you detect and track for each <class> instance throughout the video?",
    "Can you track <class> objects with varying sizes using bounding boxes?",
    "Use bounding boxes to track all <class> objects in the video and assign unique IDs to each instance.",
    "Generate bounding box coordinates for every <class> object in each frame of the video.",
    "Apply multi-object tracking with bounding boxes to follow the movement of <class> simultaneously.",
    "Provide clear bounding boxes for <class> objects to visualize their positions over time.",
    "Track <class> objects using bounding boxes and update the boxes as they change orientation.",
    "Execute MOT with bounding boxes to maintain continuous tracking of <class> for the video.",
    "Use bounding boxes to distinguish and track <class> objects with similar appearances.",
    "Generate bounding box-based tracking results for all <class> objects from start to end of the video.",
    "Perform object tracking on <class> using bounding boxes and highlight the tracked instances.",
    "Carry out multi-object tracking with bounding boxes to record the trajectory of each <class> object.",
    "Provide bounding box tracking for <class> objects in complex environments within the frames.",
    "Use bounding boxes to accurately track <class> objects and report their movement patterns.",
]

ANSWERS = [
    "This is a <task>multi-object tracking (mot)</task> task. Here are the tracking results for <class> in the video.",
    "This is a <task>multi-object tracking (mot)</task> task. I provide the bounding boxes for tracked <class> objects across all frames.",
    "This is a <task>multi-object tracking (mot)</task> task. The tracking boxes for <class> in the video are presented here.",
    "This is a <task>multi-object tracking (mot)</task> task. Here are the results of tracking <class> with bounding boxes throughout the video.",
    "This is a <task>multi-object tracking (mot)</task> task. The tracking outcomes for <class> in the video sequence are as follows.",
    "This is a <task>multi-object tracking (mot)</task> task. I've generated tracking boxes for each <class> instance in the video.",
    "This is a <task>multi-object tracking (mot)</task> task. The bounding boxes for tracked <class> objects are available here.",
    "This is a <task>multi-object tracking (mot)</task> task. Here are the tracking results showing <class> objects with bounding boxes.",
    "This is a <task>multi-object tracking (mot)</task> task. The tracking boxes for <class> in each frame are provided below.",
    "This is a <task>multi-object tracking (mot)</task> task. I've applied MOT to track <class> objects with bounding boxes in the video.",
    "This is a <task>multi-object tracking (mot)</task> task. The results of tracking <class> across the video are presented here.",
    "This is a <task>multi-object tracking (mot)</task> task. Here are the bounding boxes for <class> objects tracked in real time.",
    "This is a <task>multi-object tracking (mot)</task> task. The tracking data for <class> in the video includes their bounding boxes.",
    "This is a <task>multi-object tracking (mot)</task> task. I've generated bounding boxes to track <class> objects throughout the sequence.",
    "This is a <task>multi-object tracking (mot)</task> task. The tracking results for <class> show their positions via bounding boxes.",
    "This is a <task>multi-object tracking (mot)</task> task. Here are the bounding boxes used to track <class> in each video frame.",
    "This is a <task>multi-object tracking (mot)</task> task. The MOT results for <class> include their tracked bounding boxes.",
    "This is a <task>multi-object tracking (mot)</task> task. I provide bounding box tracking for <class> objects in the video.",
    "This is a <task>multi-object tracking (mot)</task> task. The tracking outcomes for <class> are shown with bounding boxes here.",
    "This is a <task>multi-object tracking (mot)</task> task. Here are the bounding boxes that track <class> objects across the video.",
    "This is a <task>multi-object tracking (mot)</task> task. The tracking results include bounding boxes for each <class> instance in the video.",
]