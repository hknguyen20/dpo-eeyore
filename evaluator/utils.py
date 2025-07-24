import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import (
    LogitsProcessorList,
    SequenceBiasLogitsProcessor,
    ExponentialDecayLengthPenalty
)

import os
from typing import Optional, Dict, Tuple, List
def format_system_prompt(profile: Dict, indent_level: int = 0) -> List[str]:
    """
    Format profile data into system prompt lines recursively.
    
    Args:
        profile: Dictionary containing profile data
        indent_level: Current indentation level for nested items
        
    Returns:
        List of formatted lines
    """
    lines = []
    indent = "  " * indent_level
    if indent_level == 0: # Add header only at top level
        lines.extend([
            "You will act as a help-seeker struggling with negative emotions in a conversation with someone who is listening to you.",
            "\nYOUR PROFILE:"
        ])
    # Process each key-value pair
    for key, value in profile.items():
        if isinstance(value, dict):
            # For nested dictionaries, add section header and process recursively
            lines.append(f"{indent}- {key}")
            lines.extend(format_system_prompt(value, indent_level + 1))
        else:
            # For string/number values, add directly
            lines.append(f"{indent}- {key}: {value}")
    if indent_level == 0: # Add task description at the end of top level
        lines.extend([
            "\nYOUR TASK:",
            "As the client, your role is to continue the conversation by responding naturally to the supporter, reflecting the characteristics outlined in your profile."
        ])
    return lines

def format_system_prompt_str(profile: Dict) -> str:
    """Wrapper function to get the formatted system prompt as a string."""
    return "\n".join(format_system_prompt(profile))

def load_model(
    base_model_name: str,
    checkpoint_dir: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, Dict]:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_dir: Directory containing the checkpoint files. If None: load base model.
        base_model_name: Name of the base model used for training
        device: Device to load the model on
    """
  
    # Initialize model and tokenizer from base model
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # If base model have been SFT/DPO, load aved checkpoint
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "policy.pt")
        checkpoint = torch.load(checkpoint_path, map_location=device)        
        # Load the trained weights
        model.load_state_dict(checkpoint['state'])
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()

    return model, tokenizer

def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    conversation_history: List[Dict[str, str]],
    intervew_agent_question: str
) -> Optional[str]:
    """Generate a response from the model."""
    try:
        # Prepare the conversation history as a single string
        full_prompt = "\n\n".join(
            f"{entry['role']}: {entry['content']}" for entry in conversation_history
        )
        full_prompt += f"\n\nHuman: {intervew_agent_question}\n\nAssistant:"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        # Ensure padding token is set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        # EOS token generation settings from Eeyore paper
        # logits_processor = LogitsProcessorList()
        # sequence_bias = {(tokenizer.eos_token_id,): -4.0}
        # logits_processor.append(
        #     SequenceBiasLogitsProcessor(sequence_bias)
        # )
        # logits_processor.append(
        #     ExponentialDecayLengthPenalty(
        #         #TODO
        #     )
        # )
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id,
                # logits_processor=logits_processor,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        response = response[len(full_prompt):].strip()
        return response
    except ValueError as e:
        if "Input length of input_ids" in str(e):
            print(f"Skipping sample due to length exceeding max_length: {e}")
            return None
        raise e

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

def load_evaluation_prompts(depression_severity: str = "None", symptom_severity: str = "None", cognitive_distortion: str = "None") -> list:
    evaluation_prompts = []
    
    # 1-Minimal Depression, 2-Mild Depression, 3-Moderate Depression, and 4-Severe Deperession
    if depression_severity:
        return [
        f"How subtly and consistently does the client reflect a {depression_severity} level of depression in their responses?",
        ]
      
    # 1-Not exhibited, 2-Mild, 3-Moderate, and 4-Severe.
    if symptom_severity:
        symptom_severity = symptom_severity.lower()
        return [
            f"How subtly and consistently does the client reflect a {symptom_severity} level of symptoms in their responses?",
        ]
    
    # 1-Not exhibited and 2-Exhibited.
    if cognitive_distortion:
        return [
            f"How subtly and consistently does the client reflect a {cognitive_distortion} level of cognitive distortion in their responses?"
        ]


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