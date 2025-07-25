import os
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from utils import parse_conversation_to_transcript, load_evaluation_prompts, parse_evaluation_prompts, load_model, generate_response, format_system_prompt_str
from questions import Questions
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import datasets

q = Questions()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please set it in the .env file.")

def load_evaluator():
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    return client

async def evaluate_with_evaluator(client, transcript = "None", evaluation_prompts = "None", model = "gpt-4.1"):
    instruction = "You are an expert in evaluating the alignment of responses with clinical profiles. Read the provided transcript and evaluate the responses based on the given prompts. Rate on a scale of 1 to 5, where 1 is 'not aligned' and 5 is 'full alignment' with the profile." 
    output_instruction = "The output should be a list of integers in python format, each representing the evaluation score for the corresponding questions. No explanations or additional text should be included in the output."  
    input = f"{instruction}\n\n{transcript}\n\n{evaluation_prompts}\n\n{output_instruction}"
    print(input)
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": input}],
    )
    if not response or not response.choices:
        raise Exception("No response received from OpenAI")
    return response.choices[0].message.content

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
    
    try:
        if question_dimension == 'symptom_severity':
            symptom_data = profile.get('symptom severity', {})
            if not symptom_data or len(symptom_data) == 0:
                print("Empty symptom severity data in profile")
                return None
            current_symptom = next(iter(symptom_data.keys()))
        elif question_dimension == 'cognitive_distortion':
            distortion_data = profile.get('cognition distortion exhibition', {})
            if not distortion_data or len(distortion_data) == 0:
                print("Empty cognitive distortion data in profile")
                return None
            current_distortion = next(iter(distortion_data.keys()))
    except Exception as e:
        print(f"Error accessing profile data: {str(e)}")
        return None
    
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

async def evaluate_dimension(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    evaluator,
    dimension: str,
    samples: List,
    output_dir: str
) -> Dict:
    """
    Evaluate a single clinical dimension for a model.
    
    Args:
        model: The loaded model
        tokenizer: The model's tokenizer
        evaluator: OpenAI evaluator client
        dimension: Clinical dimension to evaluate
        samples: List of samples to process
        output_dir: Directory to save results
        
    Returns:
        Dict containing evaluation results for this dimension
    """
    dimension_results = []
    skipped_samples = []
    
    # Get questions for the dimension
    questions = getattr(q, dimension)
    
    for idx, sample in enumerate(samples):
        try:
            print(f"\nProcessing {dimension} - sample {idx+1}/{len(samples)}...")
            profile = json.loads(sample['profile'])
            
            # Run interview
            conversation = run_interview_session(
                model=model,
                tokenizer=tokenizer,
                questions=questions,
                question_dimension=dimension,
                profile=profile
            )
            
            if conversation is None:
                print(f"Skipping sample {idx}: Missing required profile data")
                skipped_samples.append(idx)
                continue
                
            if not conversation or len(conversation) <= 1:
                print(f"Skipping sample {idx}: No valid responses")
                skipped_samples.append(idx)
                continue
                
            # Create and save transcript
            transcript = parse_conversation_to_transcript(
                conversation,
                speaker_labels={'user': 'Supporter', 'assistant': 'Patient', 'system': 'System'},
                include_system=True
            )
            
            # Get severity level
            if dimension == "depression_severity":
                severity = profile.get('depression severity', '').split('-')[0]
                if not severity:
                    raise ValueError("Empty depression severity value")
            elif dimension == "symptom_severity":
                symptom_data = profile.get('symptom severity', {})
                if not symptom_data:
                    raise ValueError("Empty symptom severity data")
                symptom = next(iter(symptom_data.keys()))
                severity = symptom_data[symptom].split('-')[0]
            else:  # cognitive_distortion
                distortion_data = profile.get('cognition distortion exhibition', {})
                if not distortion_data:
                    raise ValueError("Empty cognitive distortion data")
                distortion = next(iter(distortion_data.keys()))
                severity = distortion_data[distortion].split('-')[1]
                
            # Evaluate responses
            evaluation_prompts = load_evaluation_prompts(**{dimension: severity})
            evaluation = await evaluate_with_evaluator(
                evaluator,
                transcript=transcript,
                evaluation_prompts=parse_evaluation_prompts(evaluation_prompts)
            )
            
            score = json.loads(evaluation)[0] if not isinstance(evaluation, list) else evaluation[0]
            
            # Save individual results
            result = {
                'sample_id': idx,
                'severity': severity,
                'evaluation_score': score
            }
            dimension_results.append(result)
            
            # Save transcript
            os.makedirs(f"{output_dir}/transcripts", exist_ok=True)
            with open(f"{output_dir}/transcripts/{dimension}_sample_{idx}.txt", 'w') as f:
                f.write(transcript)
                
        except Exception as e:
            print(f"Error processing {dimension} sample {idx}: {str(e)}")
            skipped_samples.append(idx)
            continue
    
    # Compute dimension summary
    if dimension_results:
        scores = [r['evaluation_score'] for r in dimension_results]
        dimension_summary = {
            'dimension': dimension,
            'samples_processed': len(dimension_results),
            'samples_skipped': len(skipped_samples),
            'skipped_sample_ids': skipped_samples,
            'average_score': round(sum(scores) / len(scores), 2),
            'min_score': min(scores),
            'max_score': max(scores)
        }
    else:
        dimension_summary = {
            'dimension': dimension,
            'error': 'No valid results'
        }
        
    return dimension_summary

async def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: List,
    model_name: str,
    checkpoint_dir: str,
) -> Dict:
    """
    Evaluate a model across all clinical dimensions asynchronously.
    
    Args:
        model: Loaded model
        tokenizer: Model's tokenizer
        samples: Dataset samples to evaluate
        model_name: Name of the model (for logging)
    Returns:
        Dict with evaluation results
    """
    # Setup
    evaluator = load_evaluator()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join("evaluation", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Define dimensions to evaluate
    dimensions = ["depression_severity", "symptom_severity", "cognitive_distortion"]
    
    # Run evaluations for each dimension concurrently
    tasks = [
        evaluate_dimension(
            model=model,
            tokenizer=tokenizer,
            evaluator=evaluator,
            dimension=dimension,
            samples=samples,
            output_dir=output_dir
        )
        for dimension in dimensions
    ]
    
    # Wait for all dimensions to complete
    dimension_summaries = await asyncio.gather(*tasks)
    
    # Compile final summary
    final_summary = {
        'model_name': model_name,
        'checkpoint_dir': checkpoint_dir,
        'evaluation_timestamp': timestamp,
        'num_samples': len(samples),
        'dimension_results': dimension_summaries
    }
    
    # Save final summary
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(final_summary, f, indent=2)
        
    return final_summary

def main():
    
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

async def main():
    # Configuration
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    checkpoint_dir = "/home/20nguyen.hk/dpo-eeyore/.cache/20nguyen.hk/sft-eeyore_2025-07-20_08-07-47_636201/LATEST"
    # checkpoint_dir = None
    num_samples = 10
    dataset_name = 'liusiyang/eeyore_profile'
    split = 'train'
    
    print("\nLoading model and dataset...")
    # Load model once
    model, tokenizer = load_model(model_name, checkpoint_dir)
    
    # Load dataset once
    dataset = datasets.load_dataset(dataset_name, split=split)
    samples = dataset.select(range(num_samples))
    
    print(f"\nStarting evaluation of {model_name}")
    print(f"Number of samples: {num_samples}")
    
    summary = await evaluate_model(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        model_name=model_name,
        checkpoint_dir=checkpoint_dir
    )
    
    print("\nEvaluation Complete!")
    output_path = os.path.join("evaluation", summary['evaluation_timestamp'], "summary.json")
    print(f"Summary saved to: {output_path}")
    print("\nResults by dimension:")
    for dim_result in summary['dimension_results']:
        print(f"\n{dim_result['dimension']}:")
        print(f"  Samples processed: {dim_result['samples_processed']}")
        if 'average_score' in dim_result:
            print(f"  Average score: {dim_result['average_score']}")
            print(f"  Score range: {dim_result['min_score']} - {dim_result['max_score']}")

if __name__ == "__main__":
    asyncio.run(main())