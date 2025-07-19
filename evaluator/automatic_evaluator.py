import os
from dotenv import load_dotenv

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


def load_the_evaluation_prompts(type_of_clinically_relevant: str = "None", symptom: str = "None", cognitive_distortion: str = "None") -> list:
    
    symptom_severity = symptom.split("_")[-1] if symptom != "None" else "None" 
    
    if type_of_clinically_relevant == "None":
        raise ValueError("Please specify the type of clinically relevant prompts.")
    # 1-Minimal Depression, 2-Mild Depression, 3-Moderate Depression, and 4-Severe Deperession
    if type_of_clinically_relevant == "depression_severity":
        prompts = [
            f""
        ]
        return prompts
    # 1-Not exhibited, 2-Mild, 3-Moderate, and 4-Severe.
    elif type_of_clinically_relevant == "symptom_severity":
        if symptom_severity == "None":
            raise ValueError("Please specify the symptom_severity.")
        prompts = [
            f""
        ]
        return prompts
    # 1-Not exhibited and 2-Exhibited.
    elif type_of_clinically_relevant == "cognitive_distortion":
        if cognitive_distortion == "None":
            raise ValueError("Please specify the cognitive distortion.")
        prompts = [
            f""
        ]
        return prompts
    else:
        raise ValueError("Invalid type of clinically relevant prompts. Please choose from 'depression_severity', 'symptom_severity', or 'cognitive_distortion'.")
    

def load_evaluator():
    from openai import OpenAI
    client = OpenAI()
    return client

def evaluate_with_evaluator(client, transcript = "None", evaluation_prompts = "None", model = "gpt-4.1"):
    response = client.responses.create(
        model= model,
        input="Write a one-sentence bedtime story about a unicorn."
    )
    if response.error:
        raise Exception(f"Error in response: {response.error}")
    return response.output_text

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
    
    ## CODE HERE ##
    # Load the responses (This should be replaced with actual data loading logic)
    
    responses = [
     # CALL MODEL HERE TO GENERATE RESPONSES
    ]
    
    
    # Save responses into a file (This should be replaced with actual file saving logic)
    with open('responses.txt', 'w') as f:
        for response in responses:
            f.write(response + '\n')

    
    ## CODE HERE ## 
    # Use gpt-4.1 to evaluate the alignment (This should be replaced with actual evaluation logic)
    evaluator = load_evaluator()
    
    evaluations = [5, 4, 3]  # Dummy evaluations
    
    
    ## CODE HERE ##
    # Save the evaluation (This should be replaced with actual file saving logic)
    with open('evaluations.txt', 'w') as f:
        for eval in evaluations:
            f.write(str(eval) + '\n')
            
if __name__ == "__main__":
    main()