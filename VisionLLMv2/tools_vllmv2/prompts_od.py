"""
# prompt for GPT to write questions...
You are a helpful prompt generator. I need you to write several human-friendly and clear prompts for specific vision task with both interrogative and declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 prompts.

Task: Object Detection
Examples:
1. "In this image, could you detect all instances of <class>?",
2. "Could you please detect the objects you find that belong to the <class> category in the image?",
3. "Please perform object detection on this image for identifying <class>."
4. "I need your expertise to locate <class> in this image.",

Note: <class> are the placeholder for specific object categories. 


# prompt for GPT to write answer...
You are a helpful answer generator. I need you to write several human-friendly and clear answers for specific vision task with declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 answers.

Task: Object Detection
Examples:
1. "This is a <task>object detection (od)</task> task. Yes, here are the detection results for <class> in the image."
2. "This is a <task>object detection (od)</task> task. Certainly, the image shows the results for <class>."

Note: Do not change "This is a <task>object detection (od)</task> task.". <class> are the placeholder for specific object categories.
      Do not contain specific numbers, such as [x] instances, [x] results. 
      Do not say about the confidence scores.
"""

QUESTIONS = [
    "Can you identify all <class> objects present in this image?",
    "Please detect and mark all instances of <class> within the photograph.",
    "Could you locate any <class> objects that appear in this image?",
    "I need you to perform object detection for <class> in this scene.",
    "Would you be able to spot all <class> instances shown here?",
    "Kindly assist in detecting any <class> present within the image.",
    "Please point out every <class> object you find in this picture.",
    "Can you help me identify <class> through object detection here?",
    "Could you carry out object detection to find <class> in this frame?",
    "I'd like you to locate <class> using object detection in this image.",
    "Please analyze the image to detect any <class> objects.",
    "Would you mind identifying <class> via object detection here?",
    "Can you use object detection to find all <class> in this scene?",
    "Kindly detect every <class> instance visible in the image.",
    "Please perform object detection to highlight <class> objects here.",
    "Could you scan the image for <class> using object detection?",
    "I need your help to spot <class> through object detection in this photo.",
    "Would you be able to detect <class> within this visual content?",
    "Can you apply object detection to identify <class> in this frame?",
    "Please use object detection to locate all <class> shown here.",
    "Could you assist in finding any <class> via object detection in this image?",
]

ANSWERS = [
    "This is a <task>object detection (od)</task> task. Here are the detection results for <class> in the image, marked with bounding boxes.",
    "This is a <task>object detection (od)</task> task. The identified <class> objects in the image are highlighted clearly.",
    "This is a <task>object detection (od)</task> task. Certainly, the image shows the detected <class> instances with visual annotations.",
    "This is a <task>object detection (od)</task> task. Here's the output of the object detection for <class>, including their positions.",
    "This is a <task>object detection (od)</task> task. The results of detecting <class> in the image are presented with spatial markers.",
    "This is a <task>object detection (od)</task> task. I've located the <class> objects in the image—here are the detection findings.",
    "This is a <task>object detection (od)</task> task. The detected <class> instances in the scene are outlined for clarity.",
    "This is a <task>object detection (od)</task> task. Here are the <class> objects identified through the object detection process.",
    "This is a <task>object detection (od)</task> task. The analysis confirms the presence of <class> objects in the image, marked accordingly.",
    "This is a <task>object detection (od)</task> task. I can provide the object detection results for <class>, showing their locations.",
    "This is a <task>object detection (od)</task> task. The model has detected <class> objects in the image—here's the visual breakdown.",
    "This is a <task>object detection (od)</task> task. Here's what the object detection algorithm found for <class> in the scene.",
    "This is a <task>object detection (od)</task> task. The <class> objects detected are marked with bounding boxes in the image.",
    "This is a <task>object detection (od)</task> task. I've performed object detection for <class>, and the results are shown below.",
    "This is a <task>object detection (od)</task> task. The output includes the detected <class> objects, labeled with their positions.",
    "This is a <task>object detection (od)</task> task. Here are the <class> entities found in the image via the object detection pipeline.",
    "This is a <task>object detection (od)</task> task. The object detection process identified <class> objects in the scene, marked clearly.",
    "This is a <task>object detection (od)</task> task. I can share the detailed results of <class> detection, including their locations.",
    "This is a <task>object detection (od)</task> task. The following <class> objects were detected in the image with visual annotations.",
    "This is a <task>object detection (od)</task> task. Here's the outcome of the object detection model for <class>, showing all identified instances.",
    "This is a <task>object detection (od)</task> task. The <class> detections in the image are presented with bounding boxes for reference.",
]