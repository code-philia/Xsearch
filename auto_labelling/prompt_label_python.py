import json
from transformers import AutoTokenizer
from openai import OpenAI
import httpx
import os
import argparse
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
from io import StringIO
import tokenize
import re
from typing import Tuple, List, Dict, Any

def remove_comments_and_docstrings(source,lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp=[]
        for x in out.split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp=[]
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)

def extract_json_from_code_block(response_text: str) -> str:
    response_text = response_text.strip()
    
    if response_text.startswith('```json') and response_text.endswith('```'):
        json_content = response_text[7:-3].strip()
        return json_content
    elif response_text.startswith('```') and response_text.endswith('```'):
        lines = response_text.split('\n')
        if len(lines) > 2:
            json_content = '\n'.join(lines[1:-1]).strip()
            return json_content
    
    return response_text

def validate_alignment_response(response_json: str, original_code: str, original_comment: str) -> Tuple[bool, List[str]]:
    """
    Validate the JSON response of a code-comment alignment extractor.

    Checks:
      1. JSON parses and contains keys STEPWISE_DESCS, COMMENT_CONCEPTS, ALIGNMENT_MAP.
      2. Number of COMMENT_CONCEPTS >= 2.
      3. Each desc_of_step_j is assigned to at most one concept in ALIGNMENT_MAP.
      4. Each code segment in ALIGNMENT_MAP appears exactly as-is in original_code.
      5. Each word in concepts in COMMENT_CONCEPTS appears in the original_comment.

    Returns:
      (is_valid, errors)
    """
    errors: List[str] = []

    response_json = extract_json_from_code_block(response_json)

    # Parse JSON
    try:
        data = json.loads(response_json)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]

    # Check required top-level keys
    required_keys = {"STEPWISE_DESCS", "COMMENT_CONCEPTS", "ALIGNMENT_MAP"}
    missing = required_keys - data.keys()
    if missing:
        errors.append(f"Missing top-level keys: {', '.join(missing)}")

    # Validate STEPWISE_DESCS
    stepwise = data.get("STEPWISE_DESCS")
    if not isinstance(stepwise, list):
        errors.append("STEPWISE_DESCS must be a list")
        step_keys = []
    else:
        step_keys = []
        for idx, item in enumerate(stepwise):
            if not isinstance(item, dict) or len(item) != 1:
                errors.append(f"STEPWISE_DESCS[{idx}] must be a dict with one key")
                continue
            key, val = next(iter(item.items()))
            step_keys.append(key)
            if not isinstance(val, dict):
                errors.append(f"STEPWISE_DESCS[{idx}]['{key}'] must be a dict")
                continue
            if "desc" not in val or "code" not in val:
                errors.append(f"STEPWISE_DESCS[{idx}]['{key}'] missing 'desc' or 'code'")
            elif not isinstance(val["desc"], str) or not isinstance(val["code"], str):
                errors.append(f"STEPWISE_DESCS[{idx}]['{key}']['desc'] and ['code'] must be strings")

    # Normalize whitespace for text comparison
    def normalize(s: str) -> str:
        return ''.join(s.split()).lower()
    
    def tokenize_text(s: str) -> List[str]: 
        words = re.findall(r'\b\w+\b', s.lower())
        return words
    
    norm_orig_code = normalize(original_code)
    comment_words = tokenize_text(original_comment)

    # Validate COMMENT_CONCEPTS
    concepts = data.get("COMMENT_CONCEPTS")
    if not isinstance(concepts, list):
        errors.append("COMMENT_CONCEPTS must be a list")
        concept_keys = []
    else:
        concept_keys = []
        for idx, item in enumerate(concepts):
            if not isinstance(item, dict) or len(item) != 1:
                errors.append(f"COMMENT_CONCEPTS[{idx}] must be a dict with one key")
                continue
            key, val = next(iter(item.items()))
            concept_keys.append(key)
            if not isinstance(val, str):
                errors.append(f"COMMENT_CONCEPTS[{idx}]['{key}'] must be a string")
            else:
                # Check if all words in concept appear in original comment
                concept_words = tokenize_text(val)
                missing_words = []
                for word in concept_words:
                    if word not in comment_words:
                        missing_words.append(word)
                
                if missing_words:
                    errors.append(f"Concept '{val}' contains words not found in original comment: {', '.join(missing_words)}")

    # Number of concepts >= 2
    if len(concept_keys) < 2:
        errors.append(f"Expected at least 2 concepts, found {len(concept_keys)}")

    # Validate ALIGNMENT_MAP
    alignment = data.get("ALIGNMENT_MAP")
    if not isinstance(alignment, list):
        errors.append("ALIGNMENT_MAP must be a list")
    else:
        for idx, item in enumerate(alignment):
            if not isinstance(item, dict) or len(item) != 1:
                errors.append(f"ALIGNMENT_MAP[{idx}] must be a dict with one key")
                continue
            concept, pair = next(iter(item.items()))
            if concept not in concept_keys:
                errors.append(f"ALIGNMENT_MAP[{idx}] uses unknown concept '{concept}'")
            if (not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(x, str) for x in pair)):
                errors.append(f"ALIGNMENT_MAP[{idx}]['{concept}'] must be a [desc_key, code_segment] list of strings")
                continue
            desc_key, code_seg = pair
            # Check desc_key validity
            if desc_key not in step_keys:
                errors.append(f"ALIGNMENT_MAP[{idx}]['{concept}'] uses unknown step key '{desc_key}'")
            # Check code segment substring
            if normalize(code_seg) not in norm_orig_code:
                errors.append(f"Code segment '{code_seg}' not found in original code")

    is_valid = not errors
    return is_valid, errors

def strip_arg_descriptions(docstring: str) -> str:
    # Remove lines starting with `:type`, `:param`, `:rtype`, or `:return`
    cleaned_lines = []
    skip = False
    for line in docstring.splitlines():
        if re.match(r'^\s*:((type|param|rtype|return)\b)', line):
            skip = True
            continue
        # If we hit a non-indented line after skipping, stop skipping
        if skip and not re.match(r'^\s', line):
            skip = False
        if not skip and not re.match(r'^\s*:((type|param|rtype|return)\b)', line):
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def parse_args():
    parser = argparse.ArgumentParser(description='Code-Comment Alignment Extractor')
    
    parser.add_argument('--start_index', type=int, default=45000,
                        help='Starting index for processing data (default: 45000)')
    parser.add_argument('--end_index', type=int, default=50000,
                        help='Ending index for processing data (default: 50000)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    existing_data = []
    start_index = args.start_index
    end_index = args.end_index
    output_path = f"./data/output.jsonl" # your output path
    backup_path = output_path

    labeled_data = set()
    if os.path.exists(backup_path):
        with open(backup_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        existing_data.append(data)
                        if data.get("idx") is not None:
                            labeled_data.add(data["idx"])
                    except json.JSONDecodeError as e:
                        print(f"JSON parse error at line {line_num}: {e}")
                        assert False, f"JSON parse error at line {line_num}: {e}"
    print(f"Read {len(existing_data)} processed entries")

    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    data = []
    with open("./data/train_cleaned.jsonl", "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx < start_index or line_idx in labeled_data:
                continue
            if line_idx > end_index:
                break
            
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            print(f"Processing entry {line_idx+1} (skipping first {start_index} entries)")
            lang = "python"
            raw_code = obj["original_string"]
            code = remove_comments_and_docstrings(raw_code, lang)
            raw_comment = obj["clean_docstring"]
            comment = strip_arg_descriptions(raw_comment)
            if not comment:
                continue

            initial_prompt = [
                {
                    "role": "developer",
                    "content": """
                        You are a code-comment alignment extractor. 
                        Inputs:
                          (1) comment: str
                          (2) code: str
                        Outputs (JSON only):
                          {
                            "STEPWISE_DESCS":   List["desc_of_step_j": {"desc": str, "code": str}]
                            "COMMENT_CONCEPTS": List[{"concept_i": str}],
                            "ALIGNMENT_MAP":    List[{"concept_i": Tuple["desc_of_step_j", "code segment from desc_of_step_j": str]}]
                          }
                        # Chain-of-thought steps: 
                            Step 1: Summarize the code into a step-wise description, e.g., desc_of_step_1, desc_of_step_2, and output `STEPWISE_DESCS`.
                            Step 2: Pick the `root` comment tokens, by analyzing the syntactic dependency and semantic dependency, and output `COMMENT_CONCEPTS`.
                            Step 3: Map the desc_of_step_i to concept_j, then locate the segment from desc_of_step_j, return the comment-to-code mapping as `ALIGNMENT_MAP`.
                        # Constraints: 
                           - Number of concepts >= 2
                           - ONE concept_i can have multiple relevant desc_of_step_j
                           - The function name and signature should be aligned
                           – Preserve original formatting  
                           - Output must be valid JSON, no explanations or extra keys.
                           - All words in concepts must appear in the original comment text
                    """
                },
                {
                    "role": "user",
                    "content": """
                        (1) comment: "Return a dictionary with the subset of jobs that are marked as failed"
                        (2) code: "def get_failed_jobs(self, fail_running=False, fail_pending=False): failed_jobs = {} for job_key, job_details in self.jobs.items(): if job_details.status == JobStatus.failed: failed_jobs[job_key] = job_details elif job_details.status == JobStatus.partial_failed: failed_jobs[job_key] = job_details elif fail_running and job_details.status == JobStatus.running: failed_jobs[job_key] = job_details elif fail_pending and job_details.status <= JobStatus.pending: failed_jobs[job_key] = job_details return failed_jobs"
                    """
                },
                {
                    "role": "assistant",
                    "content": """
                    {
                      "STEPWISE_DESCS": [
                        {
                          "desc_of_step_1": {
                            "desc": "Define the function signature",
                            "code": "def get_failed_jobs(self, fail_running=False, fail_pending=False):"
                          }
                        },
                        {
                          "desc_of_step_2": {
                            "desc": "Initialize an empty dictionary to collect failed jobs",
                            "code": "failed_jobs = {}"
                          }
                        },
                        {
                          "desc_of_step_3": {
                            "desc": "Iterate over all jobs in self.jobs",
                            "code": "for job_key, job_details in self.jobs.items():"
                          }
                        },
                        {
                          "desc_of_step_4": {
                            "desc": "Check if a job's status is JobStatus.failed and add it",
                            "code": "if job_details.status == JobStatus.failed:"
                          }
                        },
                        {
                          "desc_of_step_5": {
                            "desc": "Check if a job's status is JobStatus.partial_failed and add it",
                            "code": "elif job_details.status == JobStatus.partial_failed:"
                          }
                        },
                        {
                          "desc_of_step_6": {
                            "desc": "Optionally include running jobs if fail_running is True",
                            "code": "elif fail_running and job_details.status == JobStatus.running:"
                          }
                        },
                        {
                          "desc_of_step_7": {
                            "desc": "Optionally include pending jobs if fail_pending is True",
                            "code": "elif fail_pending and job_details.status <= JobStatus.pending:"
                          }
                        },
                        {
                          "desc_of_step_8": {
                            "desc": "Return the dictionary of collected failed jobs",
                            "code": "return failed_jobs"
                          }
                        }
                      ],
                      "COMMENT_CONCEPTS": [
                        {
                          "concept_1": "Return"
                        },
                        {
                          "concept_2": "dictionary"
                        },
                        {
                          "concept_3": "subset of jobs"
                        },
                        {
                          "concept_4": "marked as failed"
                        }
                      ],
                      "ALIGNMENT_MAP": [
                        {
                          "concept_1": ["desc_of_step_8", "return failed_jobs"]
                        },
                        {
                          "concept_2": ["desc_of_step_2", "failed_jobs = {}"]
                        },
                        {
                          "concept_3": ["desc_of_step_3", "self.jobs.items()"]
                        },
                        {
                          "concept_4": ["desc_of_step_4", "job_details.status == JobStatus.failed"]
                        },
                        {
                          "concept_4": ["desc_of_step_5", "job_details.status == JobStatus.partial_failed"]
                        },
                        {
                          "concept_4": ["desc_of_step_6", "fail_running and job_details.status == JobStatus.running"]
                        },
                        {
                          "concept_4": ["desc_of_step_7", "fail_pending and job_details.status <= JobStatus.pending"]
                        }
                      ]
                    }
                    """
                },
                {
                    "role": "user",
                    "content": f"""(1) comment tokens: {comment},
                        (2) code tokens: {code}"""
                },
            ]

            num_count = 0
            is_valid = False
            current_messages = initial_prompt.copy()

            response = client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=current_messages
            )
            # print(response)
            response_text = response.choices[0].message.content
            
            clean_response_text = extract_json_from_code_block(response_text)
            
            is_valid, errors = validate_alignment_response(response_text, code, comment)
            
            print("=" * 50)
            print(f"Validation result for entry {line_idx+1}: {is_valid}, errors: {errors}")
            print(f"Cleaned response for entry {line_idx+1}: {clean_response_text}")
            
            while not is_valid and num_count < 5:
                current_messages.append({
                    "role": "assistant",
                    "content": response_text
                })
                
                error_feedback = f"""
                Your previous response has validation errors. Please fix these issues and provide a corrected JSON response.

                VALIDATION ERRORS:
                {chr(10).join([f"- {error}" for error in errors])}

                REQUIREMENTS REMINDER:
                1. All code segments in ALIGNMENT_MAP must appear exactly as written in the original code
                2. All desc_keys in ALIGNMENT_MAP must exist in STEPWISE_DESCS
                3. All concept keys in ALIGNMENT_MAP must exist in COMMENT_CONCEPTS
                4. Must have at least 2 concepts in COMMENT_CONCEPTS
                5. All words in concepts in COMMENT_CONCEPTS must appear in the original comment text
                6. Output must be valid JSON format

                ORIGINAL INPUT:
                Comment: {comment}
                Code: {code}

                Please provide a corrected JSON response that addresses ALL the validation errors listed above."""

                current_messages.append({
                    "role": "user",
                    "content": error_feedback
                })
                
                response = client.chat.completions.create(
                    model="gpt-4o-2024-11-20",
                    messages=current_messages
                )
                
                response_text = response.choices[0].message.content
                clean_response_text = extract_json_from_code_block(response_text)
                is_valid, errors = validate_alignment_response(response_text, code, comment)
                num_count += 1
                
                print("=" * 50)
                print(f"Validation result for entry {line_idx+1} retry {num_count}: {is_valid}, errors: {errors}")
                print(f"Response for entry {line_idx+1} retry {num_count}: {clean_response_text[:200]}...")
            
            if not is_valid:
                print(f"Entry {line_idx+1} is still invalid after {num_count} retries, skipping")
                continue

            obj["response"] = clean_response_text
            data.append(obj)

            if len(data) % 5 == 0:
                with open(output_path, "a", encoding="utf-8") as fw:
                    for entry in data[-5:]:
                        fw.write(json.dumps(entry, ensure_ascii=False) + "\n")
                print(f"Processed {len(data)} temporal entries, saved to {output_path}")

    if data and len(data) % 5 != 0:
        remaining_count = len(data) % 5
        with open(output_path, "a", encoding="utf-8") as fw:
            for entry in data[-remaining_count:]:
                fw.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Total processed entries: {len(data)}, saved to {output_path}")