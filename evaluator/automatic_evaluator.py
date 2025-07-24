import os
from dotenv import load_dotenv
from utils import parse_conversation_to_transcript, load_evaluation_prompts, parse_evaluation_prompts, load_model, generate_response, format_system_prompt_str
import json
from  questions import Questions
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import datasets

q = Questions()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please set it in the .env file.")

def load_evaluator():
    from openai import OpenAI
    client = OpenAI()
    return client

def evaluate_with_evaluator(client, transcript = "None", evaluation_prompts = "None", model = "gpt-4.1"):
    instruction = "You are an expert in evaluating the alignment of responses with clinical profiles. Read the provided transcript and evaluate the responses based on the given prompts. Rate on a scale of 1 to 5, where 1 is 'not aligned' and 5 is 'full alignment' with the profile." 
    output_instruction = "The output should be a list of integers in python format, each representing the evaluation score for the corresponding questions. No explanations or additional text should be included in the output."  
    input = f"{instruction}\n\n{transcript}\n\n{evaluation_prompts}\n\n{output_instruction}"
    print(input)
    response = client.responses.create(
        model= model,
        input= input,
    )
    if response.error:
        raise Exception(f"Error in response: {response.error}")
    return response.output_text

def run_interview_session(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    questions: List[str],
    question_dimension: str,
    profile: Dict,
) -> List[Dict[str, str]]:
    """
    Run an interview session with the model using provided questions.
    
    Args:
        model: The loaded model
        tokenizer: The model's tokenizer
        questions: List of questions to ask
        question_dimension: Type of questions ('depression_severity', 'symptom_severity', or 'cognitive_distortion')
        profile: The client's profile data
    
    Returns:
        List of conversation turns
    """
    # Initialize conversation with system prompt
    conversation = [
        {'role': 'system', 'content': format_system_prompt_str(profile)}
    ]
    
    # For placeholder replacement
    current_symptom = None
    current_distortion = None
    
    if question_dimension == 'symptom_severity':
        current_symptom = next(iter(profile['symptom severity'].keys()))
    elif question_dimension == 'cognitive_distortion':
        current_distortion = next(iter(profile['cognition distortion exhibition'].keys()))
    
    # Ask each question
    for question in questions:
        # Replace placeholders if needed
        if '[SYMPTOM]' in question and current_symptom:
            question = question.replace('[SYMPTOM]', current_symptom)
        if '[COGNITIVE_DISTORTION]' in question and current_distortion:
            question = question.replace('[COGNITIVE_DISTORTION]', current_distortion)
            
        # Generate response
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            conversation_history=conversation,
            intervew_agent_question=question
        )
        
        # Skip this question if response generation failed
        if response is None:
            continue
            
        # Add to conversation
        conversation.extend([
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': response}
        ])
    
    return conversation

def main():
    # Load Model
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    checkpoint_dir = "/home/20nguyen.hk/dpo-eeyore/.cache/20nguyen.hk/dpo-eeyore_2025-07-20_15-58-40_995298/LATEST"
    model, tokenizer = load_model(base_model_name, checkpoint_dir)
    evaluator = load_evaluator()
    
    # Load dataset
    full_dataset = datasets.load_dataset('liusiyang/eeyore_profile', split='train')
    samples = full_dataset.select(range(2))
    
    # Select which dimension to evaluate
    dimension = "symptom_severity"  # or "symptom_severity" or "cognitive_distortion"
    
    results = []
    for idx, sample in enumerate(samples):
        print(f"\nProcessing sample {idx+1}/{len(samples)}...")
        profile = json.loads(sample['profile'])
        
        # Get questions for the selected dimension
        questions = getattr(q, dimension)
        
        # Run interview session
        conversation = run_interview_session(
            model=model,
            tokenizer=tokenizer,
            questions=questions,
            question_dimension=dimension,
            profile=profile
        )
        
        # Create transcript
        transcript = parse_conversation_to_transcript(
            conversation,
            speaker_labels={'user': 'Supporter', 'assistant': 'Patient', 'system': 'System'},
            include_system=True
        )
        
        # Get severity level from profile
        if dimension == "depression_severity":
            severity = profile['depression severity'].split('-')[0]
        elif dimension == "symptom_severity":
            symptom = next(iter(profile['symptom severity'].keys()))
            severity = profile['symptom severity'][symptom].split('-')[0]
        else:  # cognitive_distortion
            distortion = next(iter(profile['cognition distortion exhibition'].keys()))
            severity = profile['cognition distortion exhibition'][distortion].split('-')[1]
        
        # Get evaluation
        evaluation_prompts = load_evaluation_prompts(dimension, severity)
        evaluation = evaluate_with_evaluator(
            evaluator,
            transcript=transcript,
            evaluation_prompts=parse_evaluation_prompts(evaluation_prompts)
        )
        
        # Save results
        result = {
            'sample_id': idx,
            'transcript': transcript,
            'evaluation_score': json.loads(evaluation)[0] if not isinstance(evaluation, list) else evaluation[0]
        }
        results.append(result)
        
        os.makedirs('evaluations', exist_ok=True)
        # Save transcript to txt file
        with open(f'evaluations/{dimension}_sample_{idx}.txt', 'w') as f:
            f.write(transcript)

        # Save results to jsonl
        result_json = {
            'sample_id': idx,
            'evaluation_score': json.loads(evaluation)[0] if not isinstance(evaluation, list) else evaluation[0]
        }
        with open(f'evaluations/{dimension}_results.jsonl', 'a') as f:
            json.dump(result_json, f)
            f.write('\n')
            
        average_score = sum(r['evaluation_score'] for r in results) / len(results)
        summary = {
        'dimension': dimension,
        'num_samples': len(results),
        'average_score': round(average_score, 2)
        }
        
        with open(f'evaluations/{dimension}_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
            
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Dimension: {dimension}")
        print(f"Average score: {average_score:.2f}")
        print(f"Results saved to: evaluations/{dimension}_results.jsonl")
        print(f"Transcripts saved to: evaluations/{dimension}_sample_*.txt")
        print(f"Summary saved to: evaluations/{dimension}_summary.json")

if __name__ == "__main__":
    main()