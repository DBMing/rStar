import json

base_prompt = '''Please refer to the given task description and provide a thought process in the form of step-by-step pseudocode refinement.\n\nA curious user has approached you with a programming question. You should give step-by-step solutions to the user\\'s questions. For each step you can choose one of the following three actions：\n<Action 1> Defining Function Structures Using pseudocode\n<Action 2> Refine part of the pseudocode\n<Action 3> Generate python code from the pseudocode\n\n## Structure Guidelines:\n1. Please note that the pseudocode should be detailed and provide a step-by-step solution. Each step should logically build upon the previous one, moving from high-level structure to detailed implementation.\n2. Each step should follow the format: \"Step x: <Action y>...\" (where x is the step number and <Action y> is one of the specified actions).\n3. The pseudocode should be presented in the format: \"[Pseudo Start]<PSEUDOCODE>[Pseudo End]\".\n4. At the final step, provide the complete Python code in this format: \"The code is: [Code Start]<CODE>[Code End].\" Here, <CODE> should contain a working Python code based on the final pseudocode, and it must be enclosed within Python code block syntax.\n\n## Notes\n1. Aim to break down the solution into as many detailed, intermediate steps as possible while ensuring logical coherence between steps and avoiding unnecessary redundancy.\n2. The Python code solution should match the input and output requirements as described in the question. This means the solution may use terminal I/O for inputs and outputs, or it may require function parameters and return values. Carefully review the question\\'s description to determine the expected code structure, and ensure there are no input/output format errors.\n3. Gradually refine each functional part of the pseudocode, breaking down complex operations into manageable steps.\n4. Transition to Python code only once all parts of the pseudocode have been fully refined.\n5. Strictly follow the format provided in the example.\n6. Do not generate content unrelated to the answer or any other explanations.\n\n## Here is the problem description:\n\n### Description\n{instruction}\n\n### Solution\nLet's think step by step.\n'''





def process_sft_data():
    """
    data format:
    [
        {
            "prompt": "",
            "completion": ""
        },
        ...
    ]
    """
    pass

def process_dpo_data(file_path):
    """
    data format:
    [
        {
            "prompt": "",
            "chosen": "",
            "rejected": ""
        },
        ...
    ]
    """
    with open(file_path, "r") as f:
        raw_dpo_data = json.load(f)
    
    chosen_data = {}
    rejected_data = {}
    chosen_data["question"] = []
    rejected_data["question"] = []
    chosen_data["former_step_key"] = {}
    rejected_data["former_step_key"] = {}

    for ques_id, one_data in enumerate(raw_dpo_data):
        question = one_data["0"]["user_question"]
        chosen_data["question"].append(question)
        rejected_data["question"].append(question)
        former_step_key = ""
        for step_id, step_content in one_data["0"]["ost_step"].items():
            s_ = f"Step {step_id}: {step_content}\n\n"
            former_step_key += s_
        chosen_data["former_step_key"][ques_id] = {}
        rejected_data["former_step_key"][ques_id] = {}
        chosen_data["former_step_key"][ques_id][former_step_key] = []
        rejected_data["former_step_key"][ques_id][former_step_key] = []
        # chosen_data["former_step_key"][ques_id] = former_step_key
        # rejected_data["former_step_key"][ques_id] = former_step_key
        # chosen_data["last_step"][ques_id] = []
        # rejected_data["last_step"][ques_id] = []
        for last_step_content, last_step_lable in zip(one_data["0"]["last_step_content"].values(), one_data["0"]["last_step_correctness"].values()):
            if last_step_lable == True:
                chosen_data["former_step_key"][ques_id][former_step_key].append(last_step_content)
            else:
                rejected_data["former_step_key"][ques_id][former_step_key].append(last_step_content)
            
    
    
    
    
    
    
    json_chosen_data = json.dumps(chosen_data, indent=4)
    json_rejected_data = json.dumps(rejected_data, indent=4)
    
    print(json_chosen_data)
    print(json_rejected_data)
    
    with open("chosen_data.json", "w") as f:
        json.dump(chosen_data, f)
    with open("rejected_data.json", "w") as f:
        json.dump(rejected_data, f)
        
        
        
def process_dpo_data2(file_path):
    """
    data format:
    [
        {
            "prompt": "",
            "chosen": "",
            "rejected": ""
        },
        ...
    ]
    """
    DPO_DATA = []
    with open(file_path, "r") as f:
        raw_dpo_data = json.load(f)
    
    
    for one_data in raw_dpo_data:
        prompt = base_prompt.format(instruction=one_data["0"]["user_question"])
        former_step = ""
        for step_id, step_content in one_data["0"]["ost_step"].items():
            s_ = f"Step {step_id}: {step_content}\n\n"
            former_step += s_
        last_step_id = len(one_data["0"]["ost_step"])
        former_step += f"Step {last_step_id+1}: "
        chosen_data = []
        rejected_data = []
        for last_step_content, last_step_lable in zip(one_data["0"]["last_step_content"].values(), one_data["0"]["last_step_correctness"].values()):
            # print(last_step_lable)
            if last_step_lable == True:
                chosen_data.append(former_step + last_step_content)
            else:
                rejected_data.append(former_step + last_step_content)
        if len(chosen_data) == 0 or len(rejected_data) == 0:
            continue
        
        # print(len(chosen_data), len(rejected_data))
        for chosen in chosen_data:
            for rejected in rejected_data:
                dpo_data = {
                "prompt": prompt,
                "chosen": "",
                "rejected": ""
                }
                # print(repr(rejected))
                dpo_data["chosen"] = chosen
                dpo_data["rejected"] = rejected
                DPO_DATA.append(dpo_data)

    with open("dpo_data.json", "w") as f:
        json.dump(DPO_DATA, f)
        

def process_prm_data():
    """
    data format:
    [
        {
            "prompt": "",
            "response": "",
            "label": "positive/negative"
        },
        ...
    ]
    """
    pass

if __name__ == "__main__":
    
    dpo_data_file_path = "/home/pod/shared-nvme/rStar/run_outputs/last_step_record.json"
    
    process_dpo_data2(dpo_data_file_path)


