import re
import ast
from typing import Optional, Dict

class CodeSolutionParser:
    def __init__(self):
        self.steps = []
        self.final_code = None
        self.main_function = None
        
    def check_final_step(self, text: str) -> bool:
        """Check if the last step is code generation."""
        if text == "":
            return False
            
        last_step = text.lower()
        # Check if the last step mentions code generation
        code_indicators = [
            "```python"
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
    
    try:
        return run_test(test_cases, generation, debug, n_cases)
    except Exception as e:
        if debug:
            print(f"Error in running test cases: {e}")
        in_outs = test_cases
        return [-2] * len(in_outs["inputs"])
        
