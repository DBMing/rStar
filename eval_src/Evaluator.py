# Licensed under the MIT license.

from eval_src.toolkit_for_MATH.latex_answer_check import latex_answer_check as latex_equiv
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

    def _is_number(self, s) -> Tuple[bool, str]:
        try:
            res = float(s)
            return True, str(res)
        except:
            pass
        try:
            import unicodedata

            res = unicodedata.numeric(s)
            return True, str(res)
        except:
            pass
        return False, None

    def validate_completion(self, completion: str) -> bool:
        if self.answer_marker.lower() in completion.lower():
            return True

        return False

    def isolate_answer(self, text: str):
        if text is None:
            return None

        assert isinstance(text, str)
        text = text.lower()
        split_ans = text.split(self.answer_marker.lower())
        if len(split_ans) > 1:
            ans = split_ans[-1].replace(":", "").strip()
            extract_ans_temp = ans.split(".\n")[0].strip()
            if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == ".":
                extract_ans = extract_ans_temp[0:-1]
            else:
                extract_ans = extract_ans_temp
            extract_ans = extract_ans.strip().strip("\n")
            return extract_ans
        else:
            return text
        
    def find_pass_code(self, completions: List[str], user_question: str):
        if completions is None or len(completions) == 0:
            return None, None, None, None
        
        id2pass_completions = defaultdict(list)
        pass_id_list = []
        # completions.clear()
        # with open("/home/pod/shared-nvme/rStar/note.txt", "w") as f:
        #     for item in completions:
        # # 将每个元素写入文件，每个元素后面加上换行符
        #         f.write("%s\n" % item)
        
        # completions.append("The code is: [code_begin]def union_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1 + test_tup2))\r\n  return (res)[code_end]")
        # completions.append("The code is: [code_begin]def union_of_elements(tuple1, tuple2):\r\treturn tuple(set(tuple1) | set(tuple2))[code_end]")
        
        data_file_path = "/home/pod/shared-nvme/rStar/data/MBPP/test_one.json"
        with open(data_file_path, "r") as f:
            data = json.load(f)
            
        for item in data:
            if item["problem"] == user_question:
                test = item["test"]
                break
        
        for id, c in enumerate(completions):
            # print(f"-----------------------{id}")
            model_code = self.extract_answer_from_model_completion(c)
            if model_code is None:
                continue
            
            match = re.search(r'def\s+(.*?)\(', model_code)
            if match:
                function_name = match.group(1)
                # print(function_name)
            else:
                continue
            test_x = []
            pattern = r'assert\s+(\w+)\s*\('
            for t in test:
                match = re.search(pattern, t)
                test_x.append(t.replace(match.group(1), function_name))
            
            test_code = f"{model_code}\n" + "\n".join(test_x)
            pd = self.passed(test_code)
            # print(f"*********** {id} : {pd}   *********")
            if pd:
                pass_id_list.append(id)
                id2pass_completions[id].append(c)
                
        if len(pass_id_list) == 0:
            return "", completions[0], "", 0.0000001
        
        confidence = len(pass_id_list) / len(completions)  
        passed_completion = completions[pass_id_list[0]]
        return (
            "",
            passed_completion,
            pass_id_list[0],
            confidence,
        )
        
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
                pass_ratio = pass_case_count / len(correctness_results)
            else:
                pass_ratio = 0
            
            alpha = 0
            if compile_pass:
                pass_ratio = alpha * 1 + (1 - alpha) * pass_ratio
            
            print(f"*********** {id} : score  :  {pass_ratio}   *********")
            
            
            
                
        return "", completions[0], pass_ratio, solution_trace_
        

    def find_most_confident_answer(self, completions: List[str], prior_weights: List[float] = None):
        """Returns the most confident answer, its completion, its id in the input list, and its confidence."""
        if completions is None or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)
        answer2ids = defaultdict(list)
        for id, c in enumerate(completions):
            try:
                model_answer = self.extract_answer_from_model_completion(c)
                has_existed = False
                for existing_answer in answer2completions.keys():
                    if self.check_answers_equiv(model_answer, existing_answer):
                        assert not has_existed
                        has_existed = True
                        answer2completions[existing_answer].append(c)
                        answer2ids[existing_answer].append(id)
                if not has_existed:
                    answer2completions[model_answer].append(c)
                    answer2ids[model_answer].append(id)
            except:
                pass

        assert len(answer2completions.keys()) > 0, "There are no valid completions."
        if prior_weights is not None:
            assert len(completions) == len(prior_weights)
            completion2count = {}
            for answer, answer_completions in answer2completions.items():
                count = len(answer_completions)
                for answer_completion in answer_completions:
                    completion2count[answer_completion] = count

            completion2score = {}
            for id, (completion, count) in enumerate(completion2count.items()):
                prior_weight = prior_weights[id]
                score = prior_weight * (count / len(completions))
                completion2score[completion] = score

            most_confident_completion = max(completion2score.keys(), key=lambda x: completion2score[x])

            return (
                self.extract_answer_from_model_completion(most_confident_completion),
                most_confident_completion,
                completions.index(most_confident_completion),
                completion2score[most_confident_completion],
            )
        else:
            most_confident_answer = max(answer2completions.keys(), key=lambda x: len(answer2completions[x]))
            assert (
                len(answer2completions[most_confident_answer]) > 0
            ), "There are no completions for the most confident answer."
            confidence = len(answer2completions[most_confident_answer]) / len(completions)
            assert confidence > 0
            return (
                most_confident_answer,
                answer2completions[most_confident_answer][0],
                answer2ids[most_confident_answer][0],
                confidence,
            )

    def stochastic_select_answer(self, completion2score, answer2completions, completions):
        answer2score = {}
        answer_counts = {}
        for completion, score in completion2score.items():
            answer = self.extract_answer_from_model_completion(completion)
            if answer in answer2score:
                answer2score[answer] += score
                answer_counts[answer] += 1
            else:
                answer2score[answer] = score
                answer_counts[answer] = 1

        for answer in answer2score:
            answer2score[answer] /= answer_counts[answer]

        top_answers = sorted(answer2score.items(), key=lambda x: x[1], reverse=True)[:1]
        answers, scores = zip(*top_answers)
        total_score = sum(scores)
        try:
            probabilities = [score / total_score for score in scores]
            selected_answer = random.choices(answers, weights=probabilities, k=1)[0]
        except:
            selected_answer = random.choices(answers, k=1)[0]

        most_confident_completion = answer2completions[selected_answer][0]
        completion_index = completions.index(most_confident_completion)
        confidence = answer2score[selected_answer]

        return selected_answer, most_confident_completion, completion_index, confidence

    def stochastic_calculate_completion_scores(self, prior_weights, answer2completions):
        completion2count = {}
        for answer, comps in answer2completions.items():
            count = len(comps)
            for comp in comps:
                completion2count[comp] = count

        completion2score = {}
        for idx, comp in enumerate(completion2count.keys()):
            weight = prior_weights[idx] if prior_weights is not None else 1
            score = weight * completion2count[comp]
            completion2score[comp] = score
        return completion2score

    def stochastic_select_response(self, completion2score, completions):
        sorted_completions = sorted(completion2score.items(), key=lambda x: x[1], reverse=True)[:1]
        completions, scores = zip(*sorted_completions)
        total_score = sum(scores)
        try:
            probabilities = [score / total_score for score in scores]
            sampled_completion = random.choices(completions, weights=probabilities, k=1)[0]
        except:
            sampled_completion = random.choices(completions, k=1)[0]
        confidence = completion2score[sampled_completion]
        most_confident_answer = self.extract_answer_from_model_completion(sampled_completion)
        id_of_most_confident = completions.index(sampled_completion)
        return most_confident_answer, sampled_completion, id_of_most_confident, confidence

    def stochastic_find_most_confident_answer(
        self,
        completions: List[str],
        prior_weights: List[float] = None,
    ):

        if not completions or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)
        for idx, comp in enumerate(completions):
            try:
                answer = self.extract_answer_from_model_completion(comp)
                answer2completions[answer].append(comp)
            except:
                continue

        if not answer2completions:
            return None, None, None, None

        completion2score = self.stochastic_calculate_completion_scores(prior_weights, answer2completions)

        most_confident_answer, sampled_completion, id_of_most_confident, confidence = self.stochastic_select_response(
            completion2score, completions
        )
        return most_confident_answer, sampled_completion, id_of_most_confident, confidence

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        raise NotImplementedError

    def extract_answer_from_gold_solution(self, solution: str) -> str:
        raise NotImplementedError

    def extract_answer_from_model_completion(self, completion: str) -> str:
        raise NotImplementedError


class GSM8KEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        """Judge whether two answers are equivalent."""
        is_number_a, number_a = self._is_number(answer_a)
        is_number_b, number_b = self._is_number(answer_b)
        if is_number_a and is_number_b:
            correct = number_a == number_b
        else:
            correct = False

        return correct

    def extract_answer_from_gold_solution(self, solution: str | float):
        """Extract the answer from the gold solution."""
        if isinstance(solution, float):
            return str(solution)
        return solution.split("#### ")[-1].strip()

    def extract_answer_from_model_completion(self, completion: str):
        """Extract the answer from the model completion."""
        if completion is None:
            return None

        assert isinstance(completion, str)

        preds = completion
        preds = preds.split(self.answer_marker)
        answer_flag = True if len(preds) > 1 else False
        if answer_flag:
            pred = preds[1]
        else:
            pred = preds[-1]

        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

        if len(pred) == 0:
            return None
        else:
            if answer_flag:
                pred = pred[0]
            else:
                pred = pred[-1]

        if pred != "" and pred[-1] == ".":
            pred = pred[:-1]

        pred = pred.replace(",", "").replace("\n", "")
        is_number, pred = self._is_number(pred)
        if is_number:
            return pred
        else:
            return None


GSM8KHARDEvaluator = GSM8KEvaluator
MULTIARITHEvaluator = GSM8KEvaluator


class MATHEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return False

        if answer_a == "" or answer_b == "":
            return False

        answer_a = answer_a.strip()
        answer_b = answer_b.strip()

        if answer_a.lower() == answer_b.lower():
            return True

        try:
            res = latex_equiv(answer_a, answer_b)
        except Exception as e:
            print(e)
            res = False

        return res

    def extract_answer_from_gold_solution(self, solution: str):
        def remove_boxed(s):
            left = "\\boxed{"
            try:
                assert s[: len(left)] == left
                assert s[-1] == "}"
                return s[len(left) : -1]
            except:
                return None

        def last_boxed_only_string(string):
            idx = string.rfind("\\boxed")
            if idx < 0:
                idx = string.rfind("\\fbox")
                if idx < 0:
                    return None

            i = idx
            right_brace_idx = None
            num_left_braces_open = 0
            while i < len(string):
                if string[i] == "{":
                    num_left_braces_open += 1
                if string[i] == "}":
                    num_left_braces_open -= 1
                    if num_left_braces_open == 0:
                        right_brace_idx = i
                        break
                i += 1

            if right_brace_idx == None:
                retval = None
            else:
                retval = string[idx : right_brace_idx + 1]

            return retval

        return remove_boxed(last_boxed_only_string(solution))

    def extract_answer_from_model_completion(self, completion):
        answer_split = self.isolate_answer(completion)
        return answer_split


class SVAMPEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        """Judge whether two answers are equivalent."""
        is_number_a, number_a = self._is_number(answer_a)
        is_number_b, number_b = self._is_number(answer_b)
        if is_number_a and is_number_b:
            correct = number_a == number_b
        else:
            correct = False

        return correct

    def extract_answer_from_gold_solution(self, solution: str | float):
        """Extract the answer from the gold solution."""
        if isinstance(solution, float):
            return str(solution)
        return solution.strip()

    def extract_answer_from_model_completion(self, completion: str):
        """Extract the answer from the model completion."""
        if completion is None:
            return None

        assert isinstance(completion, str)

        preds = completion
        preds = preds.split(self.answer_marker)
        answer_flag = True if len(preds) > 1 else False
        if answer_flag:
            pred = preds[1]
        else:
            pred = preds[-1]

        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

        if len(pred) == 0:
            return None
        else:
            if answer_flag:
                pred = pred[0]
            else:
                pred = pred[-1]

        if pred != "" and pred[-1] == ".":
            pred = pred[:-1]

        pred = pred.replace(",", "").replace("\n", "")
        is_number, pred = self._is_number(pred)
        if is_number:
            return pred
        else:
            return None


class STGEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def _format_answer(self, answer: str):
        if answer.lower() in ["proved", "true", "yes", "correct", "positive", "affirmative", "right", "1", "t", "y"]:
            return "true"
        elif answer.lower() in ["disproved", "false", "no", "incorrect", "negative", "wrong", "0", "f", "n"]:
            return "false"
        else:
            return answer.lower()

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return False

        assert isinstance(answer_a, str) and isinstance(answer_b, str)

        format_answer_a = self._format_answer(answer_a)
        format_answer_b = self._format_answer(answer_b)
        return format_answer_a == format_answer_b or fuzz.token_sort_ratio(format_answer_a, format_answer_b) >= 90

    def extract_answer_from_gold_solution(self, solution: str):
        if solution is None:
            return None

        assert isinstance(solution, str)

        return self._format_answer(solution)

    def extract_answer_from_model_completion(self, completion: str):
        if completion is None:
            return None

        assert isinstance(completion, str)

        answer = self.isolate_answer(completion)
        if answer is None:
            return None

        return self._format_answer(answer)
    
    
    
class MBPPEvaluator(Evaluator):
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
            # print("Error occurred:", str(e), "Type:", type(e))
            result_dict['result'] = f'Error: {str(e)}'
            
    def extract_answer_from_gold_solution(self, solution: str):
        return None
            
    def extract_answer_from_model_completion(self, completion: str):
        if completion is None:
            return None
        
        assert isinstance(completion, str)
        
        preds = completion.replace('\\n', '\n')
        code_maker = "The code is: \[Code Start\]\s*(.*?)\s*\[Code End\]"
        # code_maker = r'The code is:\s*\[Code Start\]\s*```python\s*(.*?)\s*```\s*\[Code End\]'
        # code_maker = r'\[Code Start\]\s*```python\s*(.*?)\s*```\s*\[Code End\]'
        code = re.search(code_maker, preds, re.DOTALL)
        
        if code:
            # print("Code found.")
            result = code.group(1)
            # print(str(result.replace('\\r\\n', '\n').replace('\\t', '\t').replace('\\r', '\n')))
            # return str(result)
            return str(result.replace('\\r', '').replace('\\n', '\n').replace('\\t', '\t'))
        else:
            # print("No match found.")
            return None
        
class MBPPPLUSEvaluator(Evaluator):
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
            # print("Error occurred:", str(e), "Type:", type(e))
            result_dict['result'] = f'Error: {str(e)}'
            
    def extract_answer_from_gold_solution(self, solution: str):
        return None
            
    def extract_answer_from_model_completion(self, completion: str):
        if completion is None:
            return None
        
        assert isinstance(completion, str)
        
        preds = completion
        code_maker = "The code is: \[Code Start\](.*?)\[Code End\]"
        code = re.search(code_maker, preds, re.DOTALL)
        
        if code:
            # print("Code found.")
            result = code.group(1)
            # print(str(result.replace('\\r\\n', '\n').replace('\\t', '\t').replace('\\r', '\n')))
            return str(result.replace('\\r\\n', '\n').replace('\\t', '\t').replace('\\r', '\n'))
        else:
            # print("No match found.")
            return None
        
        
        
    
    
    
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
            # print("Error occurred:", str(e), "Type:", type(e))
            result_dict['result'] = f'Error: {str(e)}'
            
    def extract_answer_from_gold_solution(self, solution: str):
        return None
            
    def extract_answer_from_model_completion(self, completion: str):
        if completion is None:
            return None
        
        assert isinstance(completion, str)
        
        preds = completion.replace('\\n', '\n')
        code_maker = "The code is: \[Code Start\]\s*(.*?)\s*\[Code End\]"
        # code_maker = r'The code is:\s*\[Code Start\]\s*```python\s*(.*?)\s*```\s*\[Code End\]'
        # code_maker = r'\[Code Start\]\s*```python\s*(.*?)\s*```\s*\[Code End\]'
        code = re.search(code_maker, preds, re.DOTALL)
        
        if code:
            # print("Code found.")
            result = code.group(1)
            # print(str(result.replace('\\r\\n', '\n').replace('\\t', '\t').replace('\\r', '\n')))
            # return str(result)
            return str(result.replace('\\r', '').replace('\\n', '\n').replace('\\t', '\t'))
        else:
            # print("No match found.")
            return None