"""
# prompt for GPT to write questions...
You are a helpful prompt generator. I need you to write several human-friendly and clear prompts for specific vision task with both interrogative and declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 prompts.

Task: Instance Segmentation
Examples:
1. "In this image, could you segment all instances of <class>?",
2. "Could you please segment the objects you find that belong to the <class> category in the image?",
3. "Please perform instance segmentation on this image for identifying <class>."
4. "I need your expertise to provide masks for <class> in this image.",

Note: <class> are the placeholder for specific object categories. 


# prompt for GPT to write answer...
You are a helpful answer generator. I need you to write several human-friendly and clear answers for specific vision task with declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 answers.

Task: Instance Segmentation
Examples:
1. "This is a <task>instance segmentation (is)</task> task. Here are the segmentation results for <class> in the image."
2. "This is a <task>instance segmentation (is)</task> task. The image shows the segmentation masks for <class>."

Note: Do not change "This is a <task>instance segmentation (is)</task> task.". <class> are the placeholder for specific object categories.
      Do not contain specific numbers, such as [x] instances, [x] results. 
      Do not say about the confidence scores.
"""

QUESTIONS = [
    "Can you segment each instance of <class> present in this image?",
    "Please provide individual masks for every <class> object in the picture.",
    "Could you identify and segment each <class> instance separately?",
    "I need you to perform instance segmentation on all <class> objects here.",
    "Would you mind generating masks for each <class> in the image?",
    "Kindly segment out every <class> instance you detect in this photo.",
    "Can you outline each <class> object with a distinct mask?",
    "Please carry out instance segmentation for <class> in this scene.",
    "Could you isolate each <class> instance via segmentation?",
    "I request masks for every single <class> in the image.",
    "Would you be able to segment all <class> instances clearly?",
    "Please mark each <class> object with a separate segmentation mask.",
    "Can you perform instance segmentation to separate <class> objects?",
    "I need your help to generate masks for each <class> here.",
    "Could you outline every <class> instance with mask individually?",
    "Kindly provide segmentation for each <class> in the scene.",
    "Would you segment each <class> object with a distinct boundary?",
    "Please generate masks to separate all <class> instances.",
    "Can you identify and mask each <class> individually?",
    "I need instance segmentation masks for every <class> in the image.",
]

ANSWERS = [
    "This is a <task>instance segmentation (is)</task> task. Here are the segmentation results for <class> in the image.",
    "This is a <task>instance segmentation (is)</task> task. The image displays the segmentation masks for <class>.",
    "This is a <task>instance segmentation (is)</task> task. The generated outputs include masks for each <class> instance.",
    "This is a <task>instance segmentation (is)</task> task. Each <class> object has been individually segmented in the result.",
    "This is a <task>instance segmentation (is)</task> task. The segmentation masks for <class> are provided here.",
    "This is a <task>instance segmentation (is)</task> task. The <class> objects are outlined with distinct segments.",
    "This is a <task>instance segmentation (is)</task> task. The results show clear segmentation of <class> instances.",
    "This is a <task>instance segmentation (is)</task> task. The <class> objects have been separately masked.",
    "This is a <task>instance segmentation (is)</task> task. Here, you can find the segmentations for each <class> in the image.",
    "This is a <task>instance segmentation (is)</task> task. The output includes individual masks for <class> objects.",
    "This is a <task>instance segmentation (is)</task> task. The <class> instances are marked with distinct segmentation boundaries.",
    "This is a <task>instance segmentation (is)</task> task. The segmentation of <class> is presented in the results.",
    "This is a <task>instance segmentation (is)</task> task. The <class> objects have been segmented as requested.",
    "This is a <task>instance segmentation (is)</task> task. The masks for <class> are included in the image.",
    "This is a <task>instance segmentation (is)</task> task. Here are the individual segmentations for each <class> instance.",
    "This is a <task>instance segmentation (is)</task> task. The results highlight the segmented <class> objects clearly.",
    "This is a <task>instance segmentation (is)</task> task. The <class> instances have been isolated with segmentation masks.",
    "This is a <task>instance segmentation (is)</task> task. The generated masks outline each <class> object.",
    "This is a <task>instance segmentation (is)</task> task. The segmentation of <class> is shown in the provided results.",
    "This is a <task>instance segmentation (is)</task> task. Each <class> object is marked with a separate segment.",
]