from datasets import load_dataset, load_from_disk, Dataset
import random
import json

def local_disk_to_json(from_path: str, to_path: str):
    taco = load_dataset(from_path, split='train')
    # difficulties = ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"]
    # difficulties = ["EASY", "MEDIUM", "MEDIUM_HARD"]
    difficulties = ["EASY"]
    if difficulties:
        taco = taco.filter(lambda example: example['difficulty'] in difficulties)
    from datasets import concatenate_datasets

    n_part = 500
    # indices = random.sample(range(len(taco)), n_part)
    # taco = taco.select(indices)
    
    part_taco_train = []
    random.seed(43)
    # indices = random.sample(range(len(taco)), n_part)

    def is_nested_empty_optimized(obj):
        if obj == []:
            return True
        if not isinstance(obj, list):
            return False
        for item in obj:
            if isinstance(item, list):
                if not is_nested_empty_optimized(item):
                    return False
            else:
                return False
        return True

    print(len(taco))
    kk = 1
    formatted_taco = []
    for item in taco:
        _info = {}
        for k, v in item.items():
            try:
                _info[k] = json.loads(v)
            except Exception as e:
                # warnings.warn("")
                _info[k] = v
        if _info["solutions"] == []:
            continue
        if not isinstance(_info["input_output"], dict):
            continue
        if is_nested_empty_optimized(_info["input_output"]["inputs"]):
            continue
        if is_nested_empty_optimized(_info["input_output"]["outputs"]):
            continue
        if _info["starter_code"] != '' and "fn_name" not in _info["input_output"]:
            continue
        if len(_info["input_output"]["inputs"]) != len(_info["input_output"]["outputs"]):
            continue
        if len(_info["input_output"]["inputs"]) >= 12:
            _info["input_output"]["inputs"] = _info["input_output"]["inputs"][:12]
            _info["input_output"]["outputs"] = _info["input_output"]["outputs"][:12]
        del _info["solutions"]
        del _info["starter_code"]
        del _info["raw_tags"]
        del _info["name"]
        del _info["source"]
        del _info["tags"]
        del _info["skill_types"]
        del _info["url"]
        del _info["Expected Auxiliary Space"]
        del _info["time_limit"]
        del _info["date"]
        del _info["picture_num"]
        del _info["memory_limit"]
        del _info["Expected Time Complexity"]
        
        formatted_taco.append(_info)
    
    print(len(formatted_taco))
    
    formatted_taco = random.sample(formatted_taco, n_part)
    
    # for i in range(n_part):
    #     part_taco_train.append(taco[i])
    # for i in range(n_part):
    #     _data = {}
    #     for k, v in taco[:n_part].items():
    #         try:
    #             v[i] = json.loads(v[i])
    #         except (json.JSONDecodeError, TypeError):
    #             pass
    #         _data[k] = v[i]
    #     part_taco_train.append(_data)

#     save_path = to_path
#     with open(save_path, 'w', encoding='utf-8') as f:
#         f.write(json.dumps(formatted_taco, ensure_ascii=False, indent=4))
        
local_disk_to_json(
        '/mnt/sdc1/jiangming/Project/rStar/data/TACO-REPO',
        '/mnt/sdc1/jiangming/Project/rStar/data/TACO/train_easy_medium_500.json'
    )