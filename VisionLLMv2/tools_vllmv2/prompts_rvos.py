"""
# prompt for GPT to write questions...
You are a helpful prompt generator. I need you to write several human-friendly and clear prompts for specific vision task with both interrogative and declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 prompts.

Task: Referring Video Object Segmentation
Examples:
1. "Given the video, could you produce segmentation masks for <expression> frame-by-frame?",
2. "Could you find and provide masks for <expression> along the video?",
3. "Help me track the <expression> with masks throughout the video.",
4. "Please segment and track the object referred by <expression> within the video.",

Note: <expression> are the placeholder for specific grounding expression. 


# prompt for GPT to write answer...
You are a helpful answer generator. I need you to write several human-friendly and clear answers for specific vision task with declarative sentences. In the following, I will first give the task name and some example prompts. You need to write at least 20 answers.

Task: Referring Video Object Segmentation
Examples:
1. "This is a <task>referring video object segmentation (rvos)</task> task. Here are the tracking result for <expression> in the video."
2. "This is a <task>referring video object segmentation (rvos)</task> task. I provide the results for <expression> in the outputs."

Note: Do not change "This is a <task>referring video object segmentation (rvos)</task> task.". <expression> are the placeholder for specific grounding expression. 
      Do not contain specific numbers, such as [x] instances, [x] results. 
      Do not say about the confidence scores.
"""

QUESTIONS = [
    "Can you create frame-by-frame segmentation masks for the <expression> shown in the video?",
    "I want you to generate masks that segment the <expression> accurately in each frame of the video.",
    "Could you provide the segmentation masks of the <expression> for every single frame of the video?",
    "Segment the <expression> in the video and present the corresponding masks for each frame.",
    "Would you be able to track and segment the <expression> using masks across the entire video?",
    "I need you to find and segment the <expression> in the video and give me the masks for it.",
    "Please generate segmentation masks that outline the <expression> in the video from start to finish.",
    "Can you show me the masks that segment the <expression> as it appears in the video?",
    "Track and segment the <expression> throughout the video and provide the masks for each frame.",
    "I request you to segment the <expression> in the video and provide detailed masks for every frame.",
    "Could you offer the masks that accurately segment the <expression> within the video's frames?",
    "Produce segmentation masks for the <expression> as it moves and changes in the video.",
    "Find and segment the <expression> in the video and deliver the relevant masks.",
    "Would you create masks that precisely segment the <expression> throughout the video sequence?",
    "Segment the <expression> frame by frame in the video and give me the corresponding masks.",
    "I'd like you to generate masks that segment the <expression> clearly in the provided video.",
    "Can you provide masks to segment the <expression> in the video, showing its full extent?",
    "Track the <expression> in the video and generate segmentation masks for it at every moment.",
    "Please find, segment, and provide masks for the <expression> in the given video.",
    "Segment the <expression> in the video and offer the masks that define its boundaries for each frame.",
    "Do you have the ability to create segmentation masks for the <expression> in each frame of the video?",
    "Generate masks that can segment the <expression> distinctly throughout the whole video.",
    "Could you start by segmenting the <expression> in the video and then provide the masks?",
    "Show me the masks that can effectively segment the <expression> as it evolves in the video.",
]

ANSWERS = [
    "This is a <task>referring video object segmentation (rvos)</task> task. Here are the segmentation masks for the <expression> across all frames of the video.",
    "This is a <task>referring video object segmentation (rvos)</task> task. I have segmented and tracked the <expression> in the video, and here are the corresponding results.",
    "This is a <task>referring video object segmentation (rvos)</task> task. The segmentation and tracking results for the <expression> within the video are provided below.",
    "This is a <task>referring video object segmentation (rvos)</task> task. I've generated masks that outline the <expression> throughout the video, and these are the results.",
    "This is a <task>referring video object segmentation (rvos)</task> task. The masks that segment the <expression> as it appears in the video are included in the output.",
    "This is a <task>referring video object segmentation (rvos)</task> task. Here are the results that show the segmentation of the <expression> in each frame of the video.",
    "This is a <task>referring video object segmentation (rvos)</task> task. I've tracked and segmented the <expression> throughout the video, presenting the relevant results here.",
    "This is a <task>referring video object segmentation (rvos)</task> task. The segmentation masks for the <expression> as it moves and changes in the video are given in the results.",
    "This is a <task>referring video object segmentation (rvos)</task> task. I've found and segmented the <expression> in the video, and these are the corresponding masks and tracking outcomes.",
    "This is a <task>referring video object segmentation (rvos)</task> task. The results accurately segment the <expression> in the video and provide the associated masks.",
    "This is a <task>referring video object segmentation (rvos)</task> task. Here are the masks that define the boundaries of the <expression> for each frame of the video.",
    "This is a <task>referring video object segmentation (rvos)</task> task. I've precisely segmented the <expression> throughout the video sequence and included the results.",
    "This is a <task>referring video object segmentation (rvos)</task> task. The output contains the segmentation results for the <expression> as it evolves in the video.",
    "This is a <task>referring video object segmentation (rvos)</task> task. I've created masks to segment the <expression> in the video, showing its full extent in the results.",
    "This is a <task>referring video object segmentation (rvos)</task> task. The segmentation and tracking of the <expression> in the video are completed, and here are the final results.",
    "This is a <task>referring video object segmentation (rvos)</task> task. I've generated masks that clearly segment the <expression> in the provided video, and these are the findings.",
    "This is a <task>referring video object segmentation (rvos)</task> task. The results display the masks that segment the <expression> in the video from start to finish.",
    "This is a <task>referring video object segmentation (rvos)</task> task. I've identified and segmented the <expression> in the video, presenting the masks and results here.",
    "This is a <task>referring video object segmentation (rvos)</task> task. The segmentation masks for the <expression> in the video, which define its boundaries, are shown in the results.",
    "This is a <task>referring video object segmentation (rvos)</task> task. I've tracked and segmented the <expression> frame by frame in the video, and here are all the relevant results.",
]