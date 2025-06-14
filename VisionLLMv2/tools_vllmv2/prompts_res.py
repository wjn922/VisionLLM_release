"""
# prompt for GPT to write questions...
You are a helpful prompt generator. I need you to write several human-friendly and clear prompts for specific vision task with both interrogative and declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 prompts.

Task: Referring Expression Segmentation
Examples:
1. "Help me segment <expression> in the image.",
2. "Could you provide the mask for <expression> in the image?",
3. "Find and segment the referred object of <expression> in the photo",
4. "Please segment the specific <expression> within the image.",

Note: <expression> are the placeholder for specific grounding expression. 


# prompt for GPT to write answer...
You are a helpful answer generator. I need you to write several human-friendly and clear answers for specific vision task with declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 answers.

Task: Referring Expression Segmentation
Examples:
1. "This is a <task>referring expression segmentation (res)</task> task. Here are the segmentation result for <expression> in the image."
2. "This is a <task>referring expression segmentation (res)</task> task. The image shows the results for <expression>."

Note: Do not change "This is a <task>referring expression segmentation (res)</task> task.". <expression> are the placeholder for specific grounding expression. 
      Do not contain specific numbers, such as [x] instances, [x] results. 
      Do not say about the confidence scores.
"""

QUESTIONS = [
    "Can you segment the object described by <expression> in the image?",
    "Please generate a segmentation mask for <expression> in the photo.",
    "Would you mind providing the mask for the object referred to as <expression>?",
    "I need you to locate and segment <expression> within the image.",
    "Could you outline the segmentation region corresponding to <expression> in the picture?",
    "Help me extract the segmentation of <expression> from the image.",
    "Please find the object denoted by <expression> and segment it.",
    "Can you provide the segmentation for the specific <expression> shown here?",
    "Would you be able to segment the item described by <expression> in this photo?",
    "I request you to generate a mask for <expression> in the given image.",
    "Could you detect and segment the object referred to by <expression>?",
    "Please mark and segment the region of <expression> in the image.",
    "Help me get the segmentation mask for the <expression> mentioned.",
    "Can you produce a mask for the object corresponding to <expression>?",
    "Would you please segment the <expression> specified in the image?",
    "I need the segmentation of the object described as <expression>.",
    "Could you find the <expression> and create a segmentation mask for it?",
    "Please provide the segmented region of <expression> in the photo.",
    "Help me segment out the <expression> from the image provided.",
    "Can you generate a mask to segment the <expression> in the picture?",
]

ANSWERS = [
    "This is a <task>referring expression segmentation (res)</task> task. Here is the segmentation result for <expression> in the image.",
    "This is a <task>referring expression segmentation (res)</task> task. The segmentation mask for <expression> is provided in the image.",
    "This is a <task>referring expression segmentation (res)</task> task. The results show the segmented region of <expression>.",
    "This is a <task>referring expression segmentation (res)</task> task. Here's the mask for the object corresponding to <expression>.",
    "This is a <task>referring expression segmentation (res)</task> task. The image includes the segmentation of <expression> as requested.",
    "This is a <task>referring expression segmentation (res)</task> task. The output provides the segmented area for <expression>.",
    "This is a <task>referring expression segmentation (res)</task> task. You can find the segmentation of <expression> in the result.",
    "This is a <task>referring expression segmentation (res)</task> task. The result highlights the region of <expression> through segmentation.",
    "This is a <task>referring expression segmentation (res)</task> task. Here's the segmentation result that outlines <expression>.",
    "This is a <task>referring expression segmentation (res)</task> task. The mask for <expression> is generated in the segmentation output.",
    "This is a <task>referring expression segmentation (res)</task> task. The segmentation of <expression> is shown in the provided result.",
    "This is a <task>referring expression segmentation (res)</task> task. The result depicts the segmented object for <expression>.",
    "This is a <task>referring expression segmentation (res)</task> task. Here's the region corresponding to <expression> via segmentation.",
    "This is a <task>referring expression segmentation (res)</task> task. The segmentation result marks the area of <expression> clearly.",
    "This is a <task>referring expression segmentation (res)</task> task. The generated mask corresponds to the <expression> in the image.",
    "This is a <task>referring expression segmentation (res)</task> task. The output includes the segmented <expression> as required.",
    "This is a <task>referring expression segmentation (res)</task> task. The result shows the detailed segmentation of <expression>.",
    "This is a <task>referring expression segmentation (res)</task> task. Here's the outlined region for the <expression> provided.",
    "This is a <task>referring expression segmentation (res)</task> task. The segmentation result identifies the area of <expression>.",
    "This is a <task>referring expression segmentation (res)</task> task. The mask generated here corresponds to the <expression> in the image.",

]