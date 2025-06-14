"""
# prompt for GPT to write questions...
You are a helpful prompt generator. I need you to write several human-friendly and clear prompts for specific vision task with both interrogative and declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 prompts.

Task: Single-Object Tracking
Examples:
1. "Could you track the target outlined by the given box for the video?",
2. "Please provide the tracking boxes for the targets with specified box in the first frame.",
3. "Given the annotated box for the target object, could you help track the object for the whole video?"
4. "I need your assistance in tracking the target with provided box.",


# prompt for GPT to write answer...
You are a helpful answer generator. I need you to write several human-friendly and clear answers for specific vision task with declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 answers.

Task: Single-Object Tracking
Examples:
1. "This is a <task>single-object tracking (sot)</task> task. Here are the results for <class> in the video."
2. "This is a <task>single-object tracking (sot)</task> task. I provide the tracking boxes for <class> for the video."

Note: Do not change "This is a <task>single-object tracking (sot)</task> task.". 
      Do not contain specific numbers, such as [x] instances, [x] results. 
      Do not say about the confidence scores.
"""

QUESTIONS = [
    "Can you track the object marked by the initial bounding box throughout the entire video sequence?",
    "Could you provide continuous tracking coordinates for the target specified in the first frame's box?",
    "Would you be able to generate tracking results for the target highlighted in the provided reference image?",
    "Can you help me track the moving object that's marked with a box in the starting frame?",
    "Is it possible to obtain a sequence of tracking boxes for the object identified in the initial annotation?",
    "Can you assist in tracking the object whose starting position is defined by the coordinates in this box?",
    "Would you be able to generate a tracking trajectory for the target highlighted in the initial box?",
    "Can you help me obtain tracking boxes for the object marked in the first frame's annotation?",
    "Please track the target object using the bounding box provided in the first frame of the video.",
    "Generate a sequence of tracking boxes for the object outlined in the initial annotation.",
    "Assist in continuously tracking the marked object across all frames of the input video.",
    "Provide real-time tracking results for the target specified by the given bounding box.",
    "Track the object highlighted in the reference image through the entire video sequence.",
    "Use the initial box annotation to generate a tracking trajectory for the single target.",
    "Please ensure the system accurately follows the object defined by the provided starting coordinates.",
    "Generate tracking outputs for the object marked with a blue box in the first frame.",
    "Apply single-object tracking to the target outlined in the given annotation across all frames.",
    "Provide a set of bounding boxes that track the specified object from the initial frame onward.",
]

ANSWERS = [
    "This is a <task>single-object tracking (sot)</task> task. The tracking results in the video are presented here.",
    "This is a <task>single-object tracking (sot)</task> task. I offer the tracking trajectories throughout the video.",
    "This is a <task>single-object tracking (sot)</task> task. The tracking boxes in the video are provided.",
    "This is a <task>single-object tracking (sot)</task> task. Here are the tracking outcomes in the video sequence.",
    "This is a <task>single-object tracking (sot)</task> task. The positional updates in the video are tracked and shown.",
    "This is a <task>single-object tracking (sot)</task> task. I provide the continuous tracking in the video.",
    "This is a <task>single-object tracking (sot)</task> task. The tracking process in the video is documented here.",
    "This is a <task>single-object tracking (sot)</task> task. Here are the tracked positions in each frame of the video.",
    "This is a <task>single-object tracking (sot)</task> task. The tracking bounding boxes in the video are available.",
    "This is a <task>single-object tracking (sot)</task> task. I offer the tracked path within the video.",
    "This is a <task>single-object tracking (sot)</task> task. The tracking results that follow show the location in the video.",
    "This is a <task>single-object tracking (sot)</task> task. Here are the tracking results that indicate where the object is in the video.",
    "This is a <task>single-object tracking (sot)</task> task. The tracking results in the video is presented as follows.",
    "This is a <task>single-object tracking (sot)</task> task. I provide the tracking data that shows the movement in the video.",
    "This is a <task>single-object tracking (sot)</task> task. The tracking boxes that outline the object in the video are here.",
    "This is a <task>single-object tracking (sot)</task> task. Here are the tracking results that track the object through the video.",
    "This is a <task>single-object tracking (sot)</task> task. The tracking outcomes across the video frames are provided.",
    "This is a <task>single-object tracking (sot)</task> task. I offer the tracking details that show the object's presence in the video.",
    "This is a <task>single-object tracking (sot)</task> task. The tracking results which monitor the object in the video are here.",
    "This is a <task>single-object tracking (sot)</task> task. Here are the tracking boxes that follow the object throughout the video.",
]