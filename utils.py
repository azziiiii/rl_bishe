import re


def dedent(s):
    # Split the string into lines
    lines = s.split('\n')
    
    # Find the indentation of the first non-empty line
    first_indent = None
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            # Calculate leading spaces in the first non-empty line
            first_indent = len(line) - len(stripped)
            break

    # If the string is empty or only whitespace, return it as is
    if first_indent is None:
        return s

    # Remove `first_indent` spaces from the start of each line
    new_lines = []
    for line in lines:
        # Remove only the first_indent spaces from each line, preserving relative indentation
        if line.startswith(' ' * first_indent):
            new_lines.append(line[first_indent:])
        else:
            new_lines.append(line)
    
    return '\n'.join(new_lines)


def get_code(response: str):
    """
    Extracts Python code blocks from the LLM response.

    Args:
        response (str): The full response from the LLM.

    Returns:
        Union[str, List[str], None]:
            - A string if there is only one code block.
            - A list of strings if multiple code blocks exist.
            - None if no code blocks are found.
    """
    code_pattern = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    matches = code_pattern.findall(response)
    
    if not matches:
        return None
    
    return matches[0] if len(matches) == 1 else matches

def extract_function_from_string(file_content):
    """
    Extract the only function defined in the provided Python file content string.

    Args:
        file_content (str): The Python file content as a string.

    Returns:
        callable: The extracted function object, if exactly one function is defined.

    Raises:
        ValueError: If no function or more than one function is found in the string.
    """
    namespace = {}
    try:
        exec(file_content, namespace)
    except:
        return None

    # Filter out function objects from the namespace
    functions = [value for value in namespace.values() if callable(value) and hasattr(value, "__code__")]

    if len(functions) != 1:
        # raise ValueError(f"Expected exactly one function in the string, found {len(functions)}.")
        return None

    return functions[0]

def extract_first_double_braced(text):
    start = text.find('{{')
    end = text.find('}}', start + 1)
    
    if start != -1 and end != -1:
        return text[start+2:end]
    return None
def extract_first_braced(text):
    start = text.find('{')
    end = text.find('}', start + 1)
    
    if start != -1 and end != -1:
        return text[start+1:end]
    return None

def extract_idea_description(text: str) -> str:
    """
    Extracts the idea description that starts with 'The idea of the algorithm is to'
    and ends right before the next code block (of any backtick length).
    """
    # Match any length of backtick fences (``` or `````` etc.)
    pattern = r"The idea of the algorithm is to.*?(?=(\n?`{3,}[^`]*\n)|$)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group().strip() if match else ""


def idea_distance(base_idea: str, new_idea: str) -> float:
    # Helper function to clean and split text into words
    def tokenize(text):
        return set(re.findall(r'\b\w+\b', text.lower()))
    
    base_words = tokenize(base_idea)
    new_words = tokenize(new_idea)
    
    if not new_words:
        return 0.0  # Avoid division by zero if new_idea is empty or only non-words
    
    new_introduced_words = new_words - base_words
    distance = len(new_introduced_words) / len(new_words)
    
    return distance