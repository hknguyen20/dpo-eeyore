def parse_conversation_to_transcript(conversation_data, include_system=False, speaker_labels=None):
    """
    Parse a list of conversation dictionaries into a clean transcript format.
    
    Args:
        conversation_data (list): List of dictionaries with 'role' and 'content' keys
        include_system (bool): Whether to include system messages in the transcript
        speaker_labels (dict): Custom labels for roles (e.g., {'user': 'Supporter', 'assistant': 'Client'})
    
    Returns:
        str: Formatted transcript
    """
    if not conversation_data:
        return ""
    
    # Default speaker labels
    default_labels = {
        'system': 'System',
        'user': 'User',
        'assistant': 'Assistant'
    }
    
    # Use custom labels if provided, otherwise use defaults
    labels = speaker_labels if speaker_labels else default_labels
    
    transcript_lines = []
    
    for entry in conversation_data:
        role = entry.get('role', 'unknown')
        content = entry.get('content', '').strip()
        
        # Skip empty content
        if not content:
            continue
            
        # Skip system messages if not requested
        if role == 'system' and not include_system:
            continue
        
        # Get speaker label
        speaker = labels.get(role, role.title())
        
        # Format the line
        if role == 'system':
            transcript_lines.append(f"[{speaker}] {content}")
        else:
            transcript_lines.append(f"{speaker}: {content}")
    
    return '\n\n'.join(transcript_lines)

def load_the_interview_prompts(type_of_clinically_relevant: str = "None", symptom: str = "None", cognitive_distortion: str = "None") -> list:
    """
    Load the prompts for evaluating depression severity.
    Returns a list of prompts.
    """
    SYMPTOM = symptom
    COGNITIVE_DISTORTION = cognitive_distortion
    
    if type_of_clinically_relevant == "None":
        raise ValueError("Please specify the type of clinically relevant prompts.")
    if type_of_clinically_relevant == "depression_severity":
        prompts = [ 
            "How have you been feeling emotionally over the past few weeks?", 
            "Do you still enjoy activities that you used to find fun or meaningful?", 
            "How has your energy been lately? Do you feel tired or drained often?", 
            "Do you ever feel guilty, worthless, or overly critical of yourself?", 
            "Have you had any thoughts about death, feeling hopeless, or that things won’t get better?"
        ]
        return prompts
    elif type_of_clinically_relevant == "symptom_severity":
        if SYMPTOM == "None":
            raise ValueError("Please specify the symptom.")
        prompts = [ 
            f"Have you been experiencing {SYMPTOM} recently?",
            f"How much does {SYMPTOM} affect your daily life or ability to do things you enjoy?",
            f"What, if anything, helps when {SYMPTOM} happens? Have you found ways to manage or reduce it?",
        ]
        return prompts
    elif type_of_clinically_relevant == "cognitive_distortion":
        if COGNITIVE_DISTORTION == "None":
            raise ValueError("Please specify the cognitive distortion.")
        prompts = [
            f"Can you describe a recent situation where you felt {COGNITIVE_DISTORTION} influencing your thoughts?",
            f"Have you noticed any patterns or triggers that make {COGNITIVE_DISTORTION} more frequent or intense?",
            f"What impact does {COGNITIVE_DISTORTION} have on your mood, motivation, or self-esteem?",
        ]
        return prompts
    else:
        raise ValueError("Invalid type of clinically relevant prompts. Please choose from 'depression_severity', 'symptom_severity', or 'cognitive_distortion'.")


def load_the_evaluation_prompts(depression_severity: str = "None", symptom_severity: str = "None", cognitive_distortion: str = "None") -> list:
    evaluation_prompts = []
    
    # 1-Minimal Depression, 2-Mild Depression, 3-Moderate Depression, and 4-Severe Deperession
    if depression_severity == "None":
        raise ValueError("Please specify the depression_severity.")
    prompts = [
        f"How subtly and consistently does the client reflect a {depression_severity} level of depression in their responses?",
    ]
    evaluation_prompts.extend(prompts)
    
    # 1-Not exhibited, 2-Mild, 3-Moderate, and 4-Severe.
    if symptom_severity == "None":
        raise ValueError("Please specify the symptom_severity.")
    symptom_severity = symptom_severity.lower()
    prompts = [
        f"How subtly and consistently does the client reflect a {symptom_severity} level of symptoms in their responses?",
    ]
    evaluation_prompts.extend(prompts)
    
    # 1-Not exhibited and 2-Exhibited.
    if cognitive_distortion == "None":
        raise ValueError("Please specify the cognitive distortion.")
    prompts = [
        f"How subtly and consistently does the client reflect a {cognitive_distortion} level of cognitive distortion in their responses?"
    ]
    
    
    evaluation_prompts.extend(prompts)
    return evaluation_prompts

def parse_evaluation_prompts(evaluation_prompts):
    parsed_prompts = []
    for i, prompt in enumerate(evaluation_prompts):
        parsed_prompt = f"Question {i+1}: {prompt.strip()}"
        parsed_prompts.append(parsed_prompt)
    return '\n'.join(parsed_prompts)

# Example usage with your data
# if __name__ == "__main__":
#     # Sample data structure (replace with your actual data)
#     sample_conversation = [
#         {
#             "role": "system",
#             "content": "You will act as a help-seeker struggling with negative emotions..."
#         },
#         {
#             "role": "assistant", 
#             "content": "Hello everyone, i'm not sure if this is the right sub to get an answer..."
#         },
#         {
#             "role": "user",
#             "content": "It can take up to 2 weeks for them to start properly working..."
#         }
#     ]
    
#     # Different parsing options
#     print("=== Standard Transcript ===")
#     transcript1 = parse_conversation_to_transcript(
#         sample_conversation, 
#         speaker_labels={'user': 'User', 'assistant': 'Assistant', 'system': 'System'},
#         include_system=True
#     )
#     print(transcript1)
    
#     # print("\n=== Simple Format ===")
#     # transcript2 = parse_conversation_simple(sample_conversation)
#     # print(transcript2)
    
#     # print("\n=== ML Dialogue Format ===")
#     # transcript3 = parse_conversation_for_ml(sample_conversation, "dialogue")
#     # print(transcript3)