import os
from dotenv import load_dotenv
from utils import parse_conversation_to_transcript
import json

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please set it in the .env file.")

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

def parse_evaluation_prompts(evaluation_prompts):
    parsed_prompts = []
    for i, prompt in enumerate(evaluation_prompts):
        parsed_prompt = f"Question {i+1}: {prompt.strip()}"
        parsed_prompts.append(parsed_prompt)
    return '\n'.join(parsed_prompts)

# Pipeline: 
#     1. Load the prompts (The three clinical relevant prompts)
#     2. Load the responses (The three clinical relevant responses) - this use interview prompts
#     3. Save responses into a file
#     4. Use gpt-4 to evaluate the alignment (Use 5 likert) - this use evaluation prompts
#     5. Save the evaluation 


def main():
    # Load the prompts
    depression_prompts = load_the_interview_prompts("depression_severity")
    symptom_prompts = load_the_interview_prompts("symptom_severity", symptom="anxiety")
    cognitive_prompts = load_the_interview_prompts("cognitive_distortion", cognitive_distortion="catastrophizing")
    
    # Choose the number of prompts for each type
    number_of_each_type = 1  # Number of prompts for each type
    all_prompts = depression_prompts[:number_of_each_type] + symptom_prompts[:number_of_each_type] + cognitive_prompts[:number_of_each_type]
    interview_prompts = [{"role": "assistant", 
                          "content": f"{prompt}"} for prompt in all_prompts]
    ## CODE HERE ##
    # Load the responses (This should be replaced with actual data loading logic)
    
    # responses = [
    #  # CALL MODEL HERE TO GENERATE RESPONSES
    # ]
    
    ## Example responses (This should be replaced with actual model responses)
    responses = [
        {"role": "user",
         "content": "I've been feeling really down lately, like nothing seems to matter anymore."},
    ] * len(all_prompts)  # Dummy responses for each prompt

    # Save responses into a file (This should be replaced with actual file saving logic)

    # Evaluate the responses using the evaluator
    evaluator = load_evaluator()
    evaluation_prompts = load_the_evaluation_prompts(depression_severity="mild", symptom_severity="mild", cognitive_distortion="not_exhibited")

    # Prepare the conversation data for evaluation
    conversation = [
        {'role': 'system', 
         'content': 'Clinical profile .....'},
    ]
    for i,_ in enumerate(interview_prompts):
        conversation.append(interview_prompts[i])
        conversation.append(responses[i])
    
    
    
    print(conversation)    
    transcript = parse_conversation_to_transcript(
        conversation, 
        speaker_labels={'user': 'User', 'assistant': 'Assistant', 'system': 'System'},
        include_system=True
    )
    
    
    evaluations = evaluate_with_evaluator(
        evaluator, 
        transcript=transcript, 
        evaluation_prompts=parse_evaluation_prompts(evaluation_prompts),
        model="gpt-4.1"
    )
    
    print(evaluations)
    
    try: 
        evaluations = json.loads(evaluations)
    except:
        evaluations = [-1, -1, -1]  # Default values if parsing fails
        
    
    # Save the transcript (This should be replaced with actual file saving logic)
    with open('evaluations/transcript.txt', 'w') as f:
        f.write(transcript)

    # Save the evaluation (This should be replaced with actual file saving logic)
    result = {  'depression_severity': evaluations[0],
                'symptom_severity': evaluations[1],
                'cognitive_distortion': evaluations[2]}
    
    with open('evaluations/evaluations.txt', 'w') as f:
        for key, eval in result.items():
            f.write(f"{key}: {eval}\n")

if __name__ == "__main__":
    main()