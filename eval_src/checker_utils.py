import re
import ast
from typing import Optional, Dict

class CodeSolutionParser:
    def __init__(self):
        self.steps = []
        self.final_code = None
        self.main_function = None
        
    def parse_steps(self, text: str):
        """Parse the solution text into individual steps."""
        # Split by "Step" followed by a number
        step_pattern = r'Step \d+:'
        # Get all starting positions of steps
        step_starts = [m.start() for m in re.finditer(step_pattern, text)]
        
        # Add the end of text position for the last slice
        step_starts.append(len(text))
        
        # Extract each step's content
        for i in range(len(step_starts) - 1):
            step_content = text[step_starts[i]:step_starts[i+1]].strip()
            self.steps.append(step_content)
    
    def check_final_step(self, text: str) -> bool:
        """Check if the last step is code generation."""
        if text == "":
            return False
            
        last_step = text.lower()
        # Check if the last step mentions code generation
        code_indicators = [
            "<action 3> generate python code from the pseudocode",
            "[code start]"
        ]
        
        return any(indicator in last_step for indicator in code_indicators)
    
    def extract_code(self, text: str) -> str:
        """Extract the Python code from the last step."""
        if text == "":
            return None
            
        last_step = text
        
        # Find code between triple backticks
        code_pattern = r'```python(.*?)```'
        code_match = re.search(code_pattern, last_step, re.DOTALL)
        
        if code_match:
            code = code_match.group(1).strip()
            self.final_code = code
            return code
        return None
    
    def extract_outermost_function(self) -> Optional[Dict]:
        """Extract the outermost function from the code, including class methods."""
        if not self.final_code:
            return None
            
        try:
            # Parse the code into an AST
            tree = ast.parse(self.final_code)
            
            # First try to find module-level function
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef):
                    return self._extract_function_info(node)
            
            # If no module-level function found, look for class methods
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    # Look for the first method in the class
                    for class_node in node.body:
                        if isinstance(class_node, ast.FunctionDef):
                            # Skip __init__ and other special methods
                            if not class_node.name.startswith('__'):
                                function_info = self._extract_function_info(class_node)
                                function_info['class_name'] = node.name
                                return function_info
                    
        except SyntaxError:
            return None
            
        return None
    
    def _extract_function_info(self, node: ast.FunctionDef) -> Dict:
        """Helper method to extract information from a function node."""
        function_info = {
            'name': node.name,
            'args': [arg.arg for arg in node.args.args],
            'body': ast.unparse(node)
        }
        
        # Add return type annotation if exists
        if node.returns:
            function_info['return_type'] = ast.unparse(node.returns)
            
        # Add argument type annotations if exist
        arg_types = {}
        for arg in node.args.args:
            if arg.annotation:
                arg_types[arg.arg] = ast.unparse(arg.annotation)
        if arg_types:
            function_info['arg_types'] = arg_types
        
        # Add docstring if exists
        docstring = ast.get_docstring(node)
        if docstring:
            function_info['docstring'] = docstring
            
        return function_info
    
    def process_solution(self, text: str) -> dict:
        """Process the entire solution text and return results."""
        has_code_generation = self.check_final_step(text)
        code = self.extract_code(text) if has_code_generation else None
        
        # Extract the outermost function if code exists
        main_function = None
        if code:
            main_function = self.extract_outermost_function()
        
        return {
            'has_code_generation': has_code_generation,
            'final_code': code,
            'main_function': main_function
        }
    
import json
import multiprocessing as mp
import concurrent
import numpy as np
from typing import List, Dict, Any, Union
from eval_src.testing_util import run_test

TIMEOUT = 10

def check_generation_correctness(
        test_cases: Dict[str, Union[str, List]],
        generation: str,
        timeout: int = TIMEOUT,
        debug: bool = False,
        n_cases: Optional[int] = None,
    ) -> List[bool]:
    """
    Args:
        test_cases (Dict[str, Union[str, List]]): A dictionary containing test cases with inputs and expected outputs.
        generation (str): The generated code to be tested.
        timeout (int, optional): The maximum time allowed for the test execution. Defaults to TIMEOUT.
        debug (bool, optional): If True, prints debug information. Defaults to False.
    Returns:
        List[bool]: A list of booleans indicating the correctness of each test case. If a timeout occurs, returns a list of -1s.
    """
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     try:
    #         future = executor.submit(run_test, test_cases, generation, debug, n_cases)
    #         return future.result()
    #     except concurrent.futures.TimeoutError:
    #         if debug:
    #             print("global timeout")
    #         in_outs = test_cases
    #         return [-1] * len(in_outs["inputs"])
    try:
        return run_test(test_cases, generation, debug, n_cases)
    except Exception as e:
        if debug:
            print(f"Error in running test cases: {e}")
        in_outs = test_cases
        return [-2] * len(in_outs["inputs"])
        
def eval_generation(
    generation: str,
    test_cases: Dict[str, Union[str, List]],
    debug: bool = False,
    n_cases: Optional[int] = None,
):
    def _normalize_test_result(result):

        if isinstance(result, np.ndarray):
            return result.item(0)
        if isinstance(result, np.bool_):
            return bool(result)
        return result
    try:
        judge_results = check_generation_correctness(test_cases=test_cases, generation=generation, debug=debug, n_cases=n_cases)
        if debug:
            print('[INFO]: Sucessfully run the test cases')
        fixed_judge_results = [_normalize_test_result(result) for result in judge_results]
        if any(res < 1 for res in fixed_judge_results):
            if debug:
                print('[INFO]: Code solution failed some test cases')
        return fixed_judge_results
    except Exception as e:
        import traceback
        if debug:
            print(f'[Error]: Error in running test cases: {traceback.format_exc()}')
        return [-2]

def eval_generations_parallel(
    generations: List[str],
    test_cases: Union[List, Dict[str, Union[str, List]]],
    debug: bool = False,
    n_cases: Optional[int] = None,
    return_binary: bool = True,
):
    """Evaluate multiple generations in parallel against a set of test cases.
    Args:
        generations (List[str]): A list of generated strings to be evaluated.
        test_cases (Dict[str, Union[str, List]]): A dictionary containing test cases.
        debug (bool, optional): If True, enables debug mode. Defaults to False.
        return_binary (bool, optional): If True, returns binary results (1 if all test cases pass, 0 otherwise).
                                        If False, returns the proportion of passed test cases. Defaults to True.
    Returns:
        List[Union[int, float]]: A list where each element corresponds to the evaluation result of a generation.
                                 If return_binary is True, the result is binary (1 or 0).
                                 If return_binary is False, the result is a float representing the proportion of passed test cases.
    """
    if not isinstance(test_cases, list):
        test_cases = [test_cases] * len(generations)
    eval_args = [
        (generation, test_case, debug, n_cases) for generation, test_case in zip(generations, test_cases)
    ]
    n_cores = max(1, mp.cpu_count() - 1)
    with mp.Pool(n_cores) as pool:
        eval_results = pool.starmap(eval_generation, eval_args)
    
    if return_binary:
        each_generation_passed_cases = [
            int(all(case_res > 0 for case_res in eval_res))
            for eval_res in eval_results
        ]
    else:
        each_generation_passed_cases = [
            sum(case_res > 0 for case_res in eval_res) / len(eval_res)
            for eval_res in eval_results
        ]
        
    return each_generation_passed_cases