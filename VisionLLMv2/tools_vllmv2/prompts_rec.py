"""
# prompt for GPT to write questions...
You are a helpful prompt generator. I need you to write several human-friendly and clear prompts for specific vision task with both interrogative and declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 prompts.

Task: Referring Expression Comprehension
Examples:
1. "Could you tell me where the <expression> is in the image?",
2. "Where can we locate the <expression> in the image?",
3. "Help me locate the <expression> in the image.",
4. "Please identify the <expression> within the image.",

Note: <expression> are the placeholder for specific grounding expression. 


# prompt for GPT to write answer...
You are a helpful answer generator. I need you to write several human-friendly and clear answers for specific vision task with declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 answers.

Task: Referring Expression Comprehension
Examples:
1. "This is a <task>referring expression comprehension (rec)</task> task. Here are the grounding result for <expression> in the image."
2. "This is a <task>referring expression comprehension (rec)</task> task. The image shows the results for <expression>."

Note: Do not change "This is a <task>referring expression comprehension (rec)</task> task.". <expression> are the placeholder for specific grounding expression. 
      Do not contain specific numbers, such as [x] instances, [x] results. 
      Do not say about the confidence scores.
"""

QUESTIONS = [
    "Could you find the <expression> in the image?",
    "Where exactly is the <expression> located in the picture?",
    "Please identify the position of the <expression> within the image.",
    "Can you pinpoint the <expression> shown in the scene?",
    "Help me locate the <expression> described here in the image.",
    "Would you mind showing me where the <expression> is?",
    "Kindly find the <expression> and indicate its location.",
    "I need you to determine where the <expression> is in the photo.",
    "Where can the <expression> be found in this image?",
    "Please point out the <expression> among the objects.",
    "Could you highlight the <expression> described in the image?",
    "Can you spot the <expression> as mentioned in the query?",
    "Help me find the specific <expression> in this scene.",
    "Would you locate the <expression> and mark its position?",
    "Please show me the area where the <expression> exists.",
    "I need your help to find the <expression> in the image.",
    "Where is the <expression> positioned in the given picture?",
    "Could you outline the <expression> described in the image?",
    "Kindly identify and mark the <expression> for me.",
    "Can you determine the location of the <expression> here?",
]

ANSWERS = [
    "This is a <task>referring expression comprehension (rec)</task> task. Here is the grounding result for <expression> in the image.",
    "This is a <task>referring expression comprehension (rec)</task> task. The image displays the located area for <expression>.",
    "This is a <task>referring expression comprehension (rec)</task> task. The grounding result for <expression> is marked here.",
    "This is a <task>referring expression comprehension (rec)</task> task. The position of <expression> is highlighted in the result.",
    "This is a <task>referring expression comprehension (rec)</task> task. Here, the <expression> is successfully grounded in the image.",
    "This is a <task>referring expression comprehension (rec)</task> task. The result shows the located <expression> in the scene.",
    "This is a <task>referring expression comprehension (rec)</task> task. The <expression> has been identified and marked in the image.",
    "This is a <task>referring expression comprehension (rec)</task> task. The grounding result for <expression> is provided here.",
    "This is a <task>referring expression comprehension (rec)</task> task. The <expression> is pinpointed in the generated result.",
    "This is a <task>referring expression comprehension (rec)</task> task. The image includes the grounding of <expression> as requested.",
    "This is a <task>referring expression comprehension (rec)</task> task. The result highlights the area corresponding to <expression>.",
    "This is a <task>referring expression comprehension (rec)</task> task. The <expression> has been located and marked in the output.",
    "This is a <task>referring expression comprehension (rec)</task> task. Here, you can find the grounded <expression> in the image.",
    "This is a <task>referring expression comprehension (rec)</task> task. The generated result shows the position of <expression> clearly.",
    "This is a <task>referring expression comprehension (rec)</task> task. The <expression> is successfully identified in the provided result.",
    "This is a <task>referring expression comprehension (rec)</task> task. The grounding of <expression> is included here.",
    "This is a <task>referring expression comprehension (rec)</task> task. The result marks the location of <expression> in the scene.",
    "This is a <task>referring expression comprehension (rec)</task> task. The <expression> is pinpointed in the image's result.",
    "This is a <task>referring expression comprehension (rec)</task> task. The located area for <expression> is shown in the output.",
    "This is a <task>referring expression comprehension (rec)</task> task. The grounding result clearly indicates <expression> in the image.",
]