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


def parse_conversation_simple(conversation_data, separator="\n---\n"):
    """
    Simple parser that creates a clean transcript without speaker labels.
    
    Args:
        conversation_data (list): List of dictionaries with 'role' and 'content' keys
        separator (str): Separator between messages
    
    Returns:
        str: Simple transcript with messages separated by the specified separator
    """
    messages = []
    
    for entry in conversation_data:
        content = entry.get('content', '').strip()
        role = entry.get('role', '')
        
        # Skip system messages and empty content
        if role == 'system' or not content:
            continue
            
        messages.append(content)
    
    return separator.join(messages)


# def parse_conversation_structured(conversation_data, include_metadata=True):
#     """
#     Parse conversation into a structured format with metadata.
    
#     Args:
#         conversation_data (list): List of dictionaries with 'role' and 'content' keys
#         include_metadata (bool): Whether to include role metadata
    
#     Returns:
#         str: Structured transcript
#     """
#     transcript_parts = []
#     message_count = 0
    
#     for entry in conversation_data:
#         role = entry.get('role', 'unknown')
#         content = entry.get('content', '').strip()
        
#         if not content:
#             continue
        
#         if role == 'system':
#             if include_metadata:
#                 transcript_parts.append(f"=== SYSTEM CONTEXT ===\n{content}\n")
#         else:
#             message_count += 1
#             role_label = "CLIENT" if role == "assistant" else "SUPPORTER"
            
#             if include_metadata:
#                 transcript_parts.append(f"[Message {message_count} - {role_label}]\n{content}\n")
#             else:
#                 transcript_parts.append(f"{role_label}: {content}\n")
    
#     return '\n'.join(transcript_parts)


# def parse_conversation_for_ml(conversation_data, format_type="dialogue"):
#     """
#     Parse conversation specifically for ML model input.
    
#     Args:
#         conversation_data (list): List of dictionaries with 'role' and 'content' keys
#         format_type (str): 'dialogue', 'narrative', or 'qa' format
    
#     Returns:
#         str: ML-ready transcript
#     """
#     if format_type == "dialogue":
#         # Standard dialogue format
#         lines = []
#         for entry in conversation_data:
#             role = entry.get('role', '')
#             content = entry.get('content', '').strip()
            
#             if role == 'system' or not content:
#                 continue
                
#             speaker = "User" if role == "user" else "Assistant"
#             lines.append(f"{speaker}: {content}")
        
#         return '\n\n'.join(lines)
    
#     elif format_type == "narrative":
#         # Narrative format for text generation models
#         messages = []
#         for entry in conversation_data:
#             role = entry.get('role', '')
#             content = entry.get('content', '').strip()
            
#             if role == 'system' or not content:
#                 continue
                
#             messages.append(content)
        
#         return ' '.join(messages)
    
#     elif format_type == "qa":
#         # Question-Answer pairs
#         qa_pairs = []
#         current_q = None
        
#         for entry in conversation_data:
#             role = entry.get('role', '')
#             content = entry.get('content', '').strip()
            
#             if role == 'system' or not content:
#                 continue
            
#             if role == 'user':
#                 current_q = content
#             elif role == 'assistant' and current_q:
#                 qa_pairs.append(f"Q: {current_q}\nA: {content}")
#                 current_q = None
        
#         return '\n\n'.join(qa_pairs)
    
#     return ""


# Example usage with your data
if __name__ == "__main__":
    # Sample data structure (replace with your actual data)
    sample_conversation = [
        {
            "role": "system",
            "content": "You will act as a help-seeker struggling with negative emotions..."
        },
        {
            "role": "assistant", 
            "content": "Hello everyone, i'm not sure if this is the right sub to get an answer..."
        },
        {
            "role": "user",
            "content": "It can take up to 2 weeks for them to start properly working..."
        }
    ]
    
    # Different parsing options
    print("=== Standard Transcript ===")
    transcript1 = parse_conversation_to_transcript(
        sample_conversation, 
        speaker_labels={'user': 'User', 'assistant': 'Assistant', 'system': 'System'},
        include_system=True
    )
    print(transcript1)
    
    # print("\n=== Simple Format ===")
    # transcript2 = parse_conversation_simple(sample_conversation)
    # print(transcript2)
    
    # print("\n=== ML Dialogue Format ===")
    # transcript3 = parse_conversation_for_ml(sample_conversation, "dialogue")
    # print(transcript3)