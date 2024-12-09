import json
import os

base_prompt = '''Please refer to the given task description and provide a thought process in the form of step-by-step pseudocode refinement.\n\nA curious user has approached you with a programming question. You should give step-by-step solutions to the user's questions. For each step you can choose one of the following three actions\n\n<Action 1> Defining algorithm Structures Using pseudocode\n**Description:**  \nOutline the core functions and overall structure of the solution without getting into implementation details. Define inputs, outputs, and the main tasks each function will perform.\n\n<Action 2> Refine part of the pseudocode\n**Description:**  \nAdd more details to the pseudocode, specifying the exact steps, logic, and operations each function will carry out. This prepares the pseudocode for actual coding.\n\n<Action 3> Generate python code from the pseudocode\n**Description:**  \nTranslate the refined pseudocode into executable Python code, making sure to handle inputs, outputs, and ensure correctness in the implementation.\n\n**Note:**\n- You can choose one of the three actions for each step.\n- Provide a detailed explanation of the reasoning behind each step.\n- Try to refer to the reference code as much as possible, but you can also modify it if needed (e.g. change variable names, add some comments, etc.).\n\n### Question\n{question}'''


def process_dpo_data(dpo_data_file_folder_path):
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
    chosen_data = []
    rejected_data = []    
    filenames = [f for f in os.listdir(dpo_data_file_folder_path) if f.endswith("Rollout Solutions.json")]
    
    for filename in filenames:
        with open(os.path.join(dpo_data_file_folder_path, filename), "r") as f:
            raw_dpo_data = json.load(f)
        chosen_data.clear()
        rejected_data.clear()
           
        for one_data in raw_dpo_data:
            prompt = base_prompt.format(question=one_data["trace"]["0"]["user_question"])
            former_step = ""
            for step_id, step_content in one_data["trace"]["0"]["ost_step"].items():
                s_ = f"### Step {step_id}: {step_content}\n\n"
                former_step += s_
            last_step_id = len(one_data["trace"]["0"]["ost_step"])
            
            if one_data["trace"]["0"]["ost_step_value"][str(last_step_id - 1)] != 1.0:
                rejected_data.append(former_step)
            else:
                chosen_data.append(former_step)
            
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
        

def process_prm_data(prm_data_file_folder_path):
    """
    data format:
    PRM_DATA_LABEL
    [
        {
            "prompt": "",
            "response": "",
            "label": "positive/negative"
        },
        ...
    ]
    PRM_DATA_SCORE
    [
        {
            "prompt": "",
            "response": "",
            "label": "float" # 0.0-1.0 score
        },
    ]
    """
    step_info = {}
    
    filenames = [f for f in os.listdir(prm_data_file_folder_path) if f.endswith("Complete Solutions.json")]

    # print(filenames)
    
    PRM_DATA_LABEL = []
    PRM_DATA_SCORE = []
    visted_flag = {}
    for filename in filenames:
        with open(os.path.join(prm_data_file_folder_path, filename), "r") as f:
            raw_prm_data = json.load(f)
     
        for one_data in raw_prm_data:
            pre_key = ""
            pre_step = ""
            trajectory_len = len(one_data)
            if trajectory_len not in step_info:
                step_info[trajectory_len] = 0
            step_info[trajectory_len] += 1
            for step_id, step_dict in reversed(list(one_data.items())):
                if step_id == "0":
                    prompt = base_prompt.format(question=step_dict["question"])
                    continue
                pre_key += str(step_dict["node_id"]) + ", "
                response = f"{pre_step}### Step {step_id}: {step_dict['ost_step']}"
                pre_step += f"### Step {step_id}: {step_dict['ost_step']}\n\n"
                if pre_key in visted_flag:
                    continue
                visted_flag[pre_key] = True
                prm_data_label = {
                    "prompt": prompt,
                    "response": response,
                    "label": "positive" if step_dict["step_value"] >= 0.5 else "negative"
                }
                prm_data_score = {
                    "prompt": prompt,
                    "response": response,
                    "label": step_dict["step_value"]
                }
                PRM_DATA_LABEL.append(prm_data_label)
                PRM_DATA_SCORE.append(prm_data_score)
            
    with open("prm_data_label.json", "w") as f:
        json.dump(PRM_DATA_LABEL, f)
    with open("prm_data_score.json", "w") as f:
        json.dump(PRM_DATA_SCORE, f)
    with open("step_info.json", "w") as f:
        json.dump(step_info, f)
        
def process_sft_data(sft_data_file_folder_path, sft_num = 1000):
    """
    data format:
    SFT_DATA
    [
        {
            "prompt": "",
            "completion": ""
        },
        ...
    ]
    """
    SFT_DATA = []
    pass_num = 0
    filenames = [f for f in os.listdir(sft_data_file_folder_path) if f.endswith("Rollout Solutions.json")]
    
    for filename in filenames:
        with open(os.path.join(sft_data_file_folder_path, filename), "r") as f:
            raw_sft_data = json.load(f)
        
        right_num = 0
        for one_data in raw_sft_data:
            last_step_id = len(one_data["trace"]["0"]["ost_step"])
            if one_data["trace"]["0"]["ost_step_value"][str(last_step_id - 1)] != 1.0:
                continue
            if right_num == 0:
                pass_num += 1
            right_num += 1
            if right_num > sft_num:
                break
            prompt = base_prompt.format(question=one_data["trace"]["0"]["user_question"])
            former_step = ""
            for step_id, step_content in one_data["trace"]["0"]["ost_step"].items():
                s_ = f"### Step {step_id}: {step_content}\n\n"
                former_step += s_
            
            sft_data = {
                "prompt": prompt,
                "completion": former_step
            }
            SFT_DATA.append(sft_data)
            
    with open("sft_data.json", "w") as f:
        json.dump(SFT_DATA, f)

    print(f"question_num: {len(filenames)}")
    print(f"pass_num: {pass_num}")
    print(f"pass_rate: {pass_num/len(filenames)}")


if __name__ == "__main__":
    
    synthetic_data_file_folder_path = "/root/shared-nvme/gene_data/rStar/run_outputs/TACO/Qwen2.5-Coder-14B-Instruct/test_Q_10_rollout_6---2024-12-09_19-30-39---[default]/answer_sheets"
    
    # process_prm_data(synthetic_data_file_folder_path)
    
    # process_sft_data(synthetic_data_file_folder_path, sft_num = 1000)
    
    process_dpo_data(synthetic_data_file_folder_path)


