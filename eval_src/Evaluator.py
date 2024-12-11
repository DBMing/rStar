# Licensed under the MIT license.

from eval_src.checker_utils import CodeSolutionParser, check_generation_correctness

import os, json, re
from typing import List, Dict, Tuple
from collections import defaultdict
import random
import copy
from fuzzywuzzy import fuzz, process

from multiprocessing import Manager, Process
import concurrent.futures


class Evaluator:
    def __init__(self) -> None:
        self.answer_marker = "answer is"
        self.parser = CodeSolutionParser()

        
    def find_TACO_code(self, completions: List[str], test_case: dict, solution_trace: Dict[int, Dict[str, str]],):
        if completions is None or len(completions) == 0:
            return None, None, None, None
        solution_trace_ = copy.deepcopy(solution_trace)   
        id2pass_completions = defaultdict(list)
        pass_ratio = 0
        compile_pass = False
        
        
        for id, c in enumerate(completions):
            result = self.parser.process_solution(c)
            
            generation_code = result["final_code"]
            
            if "fn_name" in test_case:
                if "main_function" in result:
                    if result["main_function"] is not None:
                        if "name" in result["main_function"]:
                            if test_case["fn_name"] != result["main_function"]['name']:
                                test_case["fn_name"] = result["main_function"]['name']
            
            
            
            if generation_code == None:
                pass_ratio = 0
                continue
            
            
            correctness_results = check_generation_correctness(test_case, generation_code, debug=False, n_cases=10)
            print(correctness_results)
            
            
            if isinstance(correctness_results, list):
                if True in correctness_results or False in correctness_results:
                    compile_pass = True
                pass_case_count = correctness_results.count(True)

                # 计算比例
                if len(correctness_results) > 0:
                    pass_ratio = pass_case_count / len(correctness_results)
                else:
                    pass_ratio = 0
            else:
                pass_ratio = 0
            
            alpha = 0
            if compile_pass:
                pass_ratio = alpha * 1 + (1 - alpha) * pass_ratio
            
            # print(f"*********** {id} : score  :  {pass_ratio}   *********")
            
            
            
                
        return "", completions[0], pass_ratio, solution_trace_
        

    
class TACOEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def passed(self, references):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            future_results = {executor.submit(self.run_code_with_timeout, references)}
            for future in concurrent.futures.as_completed(future_results):
                results.append(future.result())
        # print(results)
        return results[0] == 'passed'

    def run_code_with_timeout(self, code_string, timeout=1):
        with Manager() as manager:
            result_dict = manager.dict()
            process = Process(target=self.exec_code, args=(code_string, result_dict))
            process.start()
            process.join(timeout=timeout)
            if process.is_alive():
                process.kill()
                return "timeout"
            else:
                return result_dict['result']

    @staticmethod
    def exec_code(code, result_dict):
        result_dict['result'] = 'Not executed'
        try:
            exec_globals = {}
            exec(code, exec_globals)
            result_dict['result'] = 'passed'
        except Exception as e:
            
            result_dict['result'] = f'Error: {str(e)}'
            
    def extract_answer_from_gold_solution(self, solution: str):
        return None
            
    def extract_answer_from_model_completion(self, completion: str):
        if completion is None:
            return None
        
        assert isinstance(completion, str)
        
        preds = completion.replace('\\n', '\n')
        code_maker = "The code is: \[Code Start\]\s*(.*?)\s*\[Code End\]"
        code = re.search(code_maker, preds, re.DOTALL)
        
        if code:
            result = code.group(1)
            return str(result.replace('\\r', '').replace('\\n', '\n').replace('\\t', '\t'))
        else:
            
            return None