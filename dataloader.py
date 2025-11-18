from tree_sitter import Language, Parser
from parser import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token)
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import torch
import json
from tqdm import tqdm
import re
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from multiprocessing import Pool
import numpy as np
from functools import partial
# from transformers import RobertaTokenizer

@dataclass
class InputFeatures:
    code_tokens: List[str]
    code_ids: List[int]
    position_idx: List[int]
    dfg_to_code: List[Tuple[int, int]]
    dfg_to_dfg: List[List[int]]
    nl_tokens: List[str]
    nl_ids: List[int]
    code_url: str
    ori2cur_pos: Dict[int, Tuple[int, int]]
    concept_alignment: Optional[List[List[List[int]]]] = None
    # Add new fields for cross-sample similarity calculation
    valid_code_spans: Optional[List[Tuple[str, List[int]]]] = None  # List of (step_desc, code_span) tuples
    valid_comment_spans: Optional[List[Tuple[str, List[int]]]] = None  # List of (concept_text, comment_span) tuples

def load_parsers(lib_path: str = './parser/my-languages.so') -> Dict[str, Tuple[Parser, callable]]:
    """Load language parsers and their corresponding DFG extraction functions."""
    dfg_functions: Dict[str, callable] = {
        'python': DFG_python,
        'java': DFG_java,
        'ruby': DFG_ruby,
        'go': DFG_go,
        'php': DFG_php,
        'javascript': DFG_javascript
    }

    parsers: Dict[str, Tuple[Parser, callable]] = {}

    for lang, dfg_func in dfg_functions.items():
        language = Language(lib_path, lang)
        parser = Parser()
        parser.set_language(language)
        parsers[lang] = (parser, dfg_func)

    return parsers

def extract_dataflow(
    code: str,
    parser: Tuple[Parser, Any],  # Parser + corresponding DFG function
    lang: str
) -> Tuple[List[str], List[Tuple[str, int, str, List[int]]]]:
    """
    Remove comments, tokenize code, and extract dataflow graph (DFG).

    Args:
        code (str): The original code snippet.
        parser (Tuple[Parser, Callable]): A tuple of (tree-sitter Parser, DFG function).
        lang (str): Programming language name.

    Returns:
        Tuple[List[str], List[Tuple[str, int, str, List[int]]]]:
            - code_tokens: tokenized code strings
            - dfg: data flow graph entries, each in the format (token, index, type, [dependencies])
    """
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass

    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x,code) for x in tokens_index]
        index_to_code = {}
        for idx,(index,code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx,code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x:x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []

    return code_tokens, dfg


def normalize_token(token: str) -> str:
    """Normalize a token by stripping prefixes and lowering case."""
    return re.sub(r"^[Ġ▁]+", "", token).lower()

def normalize_and_concat(tokens: List[str]) -> str:
    """Concatenate and normalize a list of tokens into a word-like form."""
    return "".join(normalize_token(t) for t in tokens)

def shrink_nested_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    spans = sorted(spans, key=lambda x: (x[0], -x[1]))  # sort by start asc, end desc
    result = []
    for i, (s1, e1) in enumerate(spans):
        if all(
           (s1, e1) == (s2, e2) or not (s1 <= s2 and e2 <= e1) for j, (s2, e2) in enumerate(spans) if i != j
        ):
            result.append((s1, e1))
    # if nothing is nested (e.g. (1,1)), keep all
    if not result:
        return spans
    return result


def find_token_spans(subtokens: List[str], word: str, tokenizer: Any) -> List[Tuple[int, int]]:
    '''
    Match the `word` (string) against subtoken spans by normalized concat.
    Args:
        subtokens:
        word:
        tokenizer:

    Returns:
        spans:
    '''
    # Skip if word is not a string
    if not isinstance(word, str):
        return []
        
    word_norm = re.sub(r"\s+", "", word.lower())  # lowercase and remove white spaces
    norm_subtokens = [normalize_token(t) for t in subtokens]

    spans: List[Tuple[int, int]] = []
    for start in range(len(norm_subtokens)):
        concat = ""
        for end in range(start, min(len(norm_subtokens), start + 10)):  # max length 10
            concat += norm_subtokens[end]
            if concat in [tokenizer.unk_token, tokenizer.pad_token]:
                break
            if concat == word_norm:
                spans.append((start, end))
                break  # only first match per start index
    spans = shrink_nested_spans(spans)
    return spans

def normalize_token_for_matching(token: str) -> str:
    """Normalize a token by removing special markers, spaces and converting to lowercase."""
    # Remove special markers and spaces
    token = re.sub(r'[Ġ▁\s]', '', token)
    # Convert to lowercase
    return token.lower()

def find_code_span_matches(code_tokens: List[str], step_code: str, tokenizer: Any) -> List[Tuple[int, int]]:
    '''
    Find the exact span in code_tokens that matches the given step code.
    '''
    if not isinstance(step_code, str) or not step_code.strip():
        return []
    
    # 1. Preprocess step_code
    # Keep quotes, parentheses, and other special characters, only remove whitespace characters
    step_code_clean = re.sub(r'[\s\n\r\t]+', '', step_code)  # Only remove whitespace characters
    step_code_clean = step_code_clean.lower()
    
    # 2. Preprocess code_tokens
    code_text = ''
    token_positions = []
    current_pos = 0
    valid_token_indices = []
    
    for i, token in enumerate(code_tokens):
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, '<s>', '</s>']:
            continue
            
        # Only remove whitespace characters, keep other special characters
        normalized_token = normalize_token_for_matching(token)
        normalized_token = re.sub(r'[\s\n\r\t]+', '', normalized_token)
        if normalized_token:
            token_positions.append((current_pos, current_pos + len(normalized_token)))
            code_text += normalized_token
            current_pos += len(normalized_token)
            valid_token_indices.append(i)
    
    # 3. Use sliding window to match
    matches = []
    window_size = len(step_code_clean)
    
    for i in range(len(code_text) - window_size + 1):
        window = code_text[i:i + window_size]
        if window == step_code_clean:
            start_token_idx = None
            end_token_idx = None
            
            for j, (token_start, token_end) in enumerate(token_positions):
                if token_start <= i < token_end:
                    start_token_idx = valid_token_indices[j]
                if token_start <= i + window_size <= token_end:
                    end_token_idx = valid_token_indices[j]
                    break
            
            if start_token_idx is not None and end_token_idx is not None:
                matches.append((start_token_idx, end_token_idx))
    
    return matches

def build_concept_alignment(
    stepwise_descs: List[Dict[str, Dict[str, str]]],
    comment_concepts: List[Dict[str, str]],
    alignment_map: List[Dict[str, List[str]]],
    nl_tokens: List[str],
    code_tokens: List[str],
    tokenizer: Any,
    example_idx: str = "unknown"
) -> Tuple[List[List[List[int]]], List[Tuple[str, List[int]]], List[Tuple[str, List[int]]]]:
    """
    Build concept alignment and extract valid comment concepts and code steps.
    
    Returns:
        concept_alignment: List of concept alignments
        valid_code_spans: List of (step_desc, code_span) tuples for valid steps
        valid_comment_spans: List of (concept_text, comment_span) tuples for valid concepts
    """
    concept_alignment: List[List[List[int]]] = []
    step_to_spans: Dict[str, List[Tuple[int, int]]] = {}
    unmatched_steps = []
    
    # Extract comment concepts and step descriptions for cross-sample similarity
    valid_code_spans = []
    valid_comment_spans = []
    
    # 1. Build mapping from step_code to actual code spans, and store valid step descriptions and their corresponding spans
    for step in stepwise_descs:
        step_name = list(step.keys())[0]
        step_code = step[step_name].get("code")
        step_desc = step[step_name].get("desc", "")
        
        # Check if step_code is a valid string
        if not isinstance(step_code, str):
            continue
            
        # Preprocess step_code
        step_code_clean = re.sub(r'[\s\n\r\t]+', '', step_code)
        step_code_clean = step_code_clean.lower()
        # print(f"Cleaned step code: {step_code_clean}")
        
        # Try to match
        spans = find_code_span_matches(code_tokens, step_code, tokenizer)
        if spans:
            step_to_spans[step_name] = spans
            # print(f"✓ Found {len(spans)} matches for {step_name}")
            # for start, end in spans:
            #     matched_code = ' '.join(code_tokens[start:end+1])
            #     print(f"  Match: {matched_code}")
            
            # Store valid step description and corresponding code spans
            if isinstance(step_desc, str) and step_desc.strip():
                # Flatten code spans
                flat_code_spans = []
                for s, e in spans:
                    flat_code_spans.extend([s, e])
                valid_code_spans.append((step_desc.strip(), flat_code_spans))
        else:
            unmatched_steps.append({
                'step_name': step_name,
                'step_code': step_code,
                'step_desc': step_desc,
                'cleaned_code': step_code_clean
            })
            # print(f"✗ No matches found for {step_name}")

    
    # 2. Merge step_names of the same concept
    merged_concept_map: Dict[str, List[str]] = {}
    for concept_map in alignment_map:
        try:
            concept_name = list(concept_map.keys())[0]
            step_names = concept_map.get(concept_name, [])
            
            if not step_names or not isinstance(step_names, list):
                # print(f"\n=== Warning: Invalid step_names (Example {example_idx}) ===")
                # print(f"Concept name: {concept_name}")
                # print(f"Step names: {step_names}")
                # print(f"Concept map: {concept_map}")
                # print(f"Total steps in stepwise_descs: {len(stepwise_descs)}")
                # print(f"Available steps: {[list(s.keys())[0] for s in stepwise_descs]}")
                continue
                
            # Only keep strings in the format desc_of_step
            filtered_steps = [step for step in step_names if step.startswith('desc_of_step_')]
            
            if concept_name in merged_concept_map:
                merged_concept_map[concept_name].extend(filtered_steps)
            else:
                merged_concept_map[concept_name] = filtered_steps
        except (IndexError, KeyError):
            continue
    
    # 3. Process merged concept mapping, and store valid comment concepts and spans
    for concept_name, step_names in merged_concept_map.items():
        try:
            # Find all code spans corresponding to the concept, ignore unmatched steps
            code_spans = []
            for step_name in step_names:
                if step_name in step_to_spans:
                    code_spans.extend(step_to_spans[step_name])
            
            # If no code spans found, skip this concept
            if not code_spans:
                continue
            
            # Find comment spans corresponding to the concept
            try:
                concept_dict = next(c for c in comment_concepts if concept_name in c)
                concept_text = concept_dict.get("concept_1", concept_dict.get(concept_name))
                if concept_text is None:
                    # print(f"\n=== Warning: Could not find concept text (Example {example_idx}) ===")
                    # print(f"Concept name: {concept_name}")
                    # print(f"Concept dict: {concept_dict}")
                    # print(f"Available concepts: {[list(c.keys())[0] for c in comment_concepts]}")
                    # print(f"Total concepts: {len(comment_concepts)}")
                    continue
                
                # Use the original comment span search logic
                comment_words = concept_text.split()
                comment_spans = []
                for word in comment_words:
                    if len(word) == 1:
                        continue
                    word_spans = find_token_spans(nl_tokens, word, tokenizer)
                    if not word_spans:
                        # print(f"\n=== Warning: Word not found in comment (Example {example_idx}) ===")
                        # print(f"Concept name: {concept_name}")
                        # print(f"Word: {word}")
                        # print(f"Concept text: {concept_text}")
                        # print(f"Comment tokens: {nl_tokens}")
                        # print(f"Total comment tokens: {len(nl_tokens)}")
                        pass
                    comment_spans.extend(word_spans)
                
                if not comment_spans:
                    # print(f"\n=== Warning: No comment spans found (Example {example_idx}) ===")
                    # print(f"Concept name: {concept_name}")
                    # print(f"Concept text: {concept_text}")
                    # print(f"Comment words: {comment_words}")
                    # print(f"Comment tokens: {nl_tokens}")
                    # print(f"Total comment tokens: {len(nl_tokens)}")
                    continue
                
                # Flatten spans
                flat_comment_spans = []
                for s, e in comment_spans:
                    flat_comment_spans.extend([s, e])
                
                flat_code_spans = []
                for s, e in code_spans:
                    flat_code_spans.extend([s, e])
                
                concept_alignment.append([flat_comment_spans, flat_code_spans])
                
                # Store valid concept text and spans
                valid_comment_spans.append((concept_text.strip(), flat_comment_spans))
                
            except (StopIteration, KeyError) as e:
                # print(f"\n=== Warning: Error processing concept (Example {example_idx}) ===")
                # print(f"Concept name: {concept_name}")
                # print(f"Error type: {type(e).__name__}")
                # print(f"Error message: {str(e)}")
                # print(f"Available concepts: {[list(c.keys())[0] for c in comment_concepts]}")
                # print(f"Total concepts: {len(comment_concepts)}")
                continue
                
        except (IndexError, KeyError) as e:
            # print(f"\n=== Warning: Error processing concept map (Example {example_idx}) ===")
            # print(f"Concept map: {concept_map}")
            # print(f"Error type: {type(e).__name__}")
            # print(f"Error message: {str(e)}")
            # print(f"Total concepts in alignment map: {len(alignment_map)}")
            continue
    
    # Only print warning and record URL if there is no valid alignment
    if not concept_alignment:
        # Extract URL from example_idx
        url = example_idx.split(" ")[0] if isinstance(example_idx, str) else ""
        
        # Write the URL of the invalid sample to the log file
        with open("logs/invalid_alignments.txt", "a") as f:
            f.write(f"{url}\n")
    
    return concept_alignment, valid_code_spans, valid_comment_spans


class TextDataset(Dataset):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 args: object,
                 file_path: Optional[str] = None,
                 pool: Optional[Pool] = None,
                 compute_alignment: bool = True):

        self.args = args

        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                data.append((obj, tokenizer, args))

        self.examples: List[InputFeatures] = pool.map(partial(self.convert_examples_to_features, compute_alignment=compute_alignment), tqdm(data, total=len(data)))
        self.raw_code = [obj[0]["original_string"] for obj in data]
        self.raw_comment = [obj[0]["clean_docstring"] for obj in data]

        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                print("*** Example ***")
                print("idx: {}".format(idx))
                print("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
                print("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                print("position_idx: {}".format(example.position_idx))
                print("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
                print("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))
                print("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
                print("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))
                print("ori2cur_pos: {}".format(example.ori2cur_pos))
                # if compute_alignment:
                #     print("code_comment_matches: {}".format(example.concept_alignment))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[List[List[int]]], List[Tuple[str, List[int]]], List[Tuple[str, List[int]]]]:

        # calculate graph-guided masked function
        attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
                              self.args.code_length + self.args.data_flow_length), dtype=bool)

        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])

        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True

        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].code_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True

        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True

        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx):
                    attn_mask[idx + node_index, a + node_index] = True
        ori2cur_pos_list = [[start, end] for start, end in self.examples[item].ori2cur_pos.values()]

        # Set padding length
        max_len: int = self.args.code_length
        ori2cur_pos_list_padded = ori2cur_pos_list + [[0, 0]] * (max_len - len(ori2cur_pos_list))
        if len(ori2cur_pos_list_padded) > max_len:
            ori2cur_pos_list_padded = ori2cur_pos_list_padded[:max_len]

        return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(self.examples[item].nl_ids),
                torch.tensor(ori2cur_pos_list_padded),
                self.examples[item].concept_alignment,
                self.examples[item].valid_code_spans or [],
                self.examples[item].valid_comment_spans or [])

    @staticmethod
    def convert_examples_to_features(
        item: Tuple[Dict[str, Any], PreTrainedTokenizer, Any], compute_alignment: bool,
    ) -> InputFeatures:

        obj, tokenizer, args = item
        raw_code = obj["original_string"]
        raw_comment = obj["clean_docstring"]
        code_url = obj["url"]

        # Code -> tokens + ids + pos_ids + dfg
        parsers = load_parsers()
        parser = parsers[args.lang]

        # extract data flow
        code_tokens, dfg = extract_dataflow(raw_code, parser, args.lang)
        code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                       enumerate(code_tokens)]

        ori2cur_pos: Dict[int, Tuple[int, int]] = {-1: (0, 0)}
        for i in range(len(code_tokens)):
            ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
        code_tokens = [y for x in code_tokens for y in x]

        # truncating
        code_tokens = code_tokens[:args.code_length + args.data_flow_length - 2 - min(len(dfg), args.data_flow_length)]
        code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]

        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]

        dfg = dfg[:args.code_length + args.data_flow_length - len(code_tokens)]
        code_tokens += [x[0] for x in dfg]
        position_idx += [0 for x in dfg]
        code_ids += [tokenizer.unk_token_id for x in dfg]

        padding_length = args.code_length + args.data_flow_length - len(code_ids)
        position_idx += [tokenizer.pad_token_id] * padding_length
        code_ids += [tokenizer.pad_token_id] * padding_length
        code_tokens += [tokenizer.pad_token] * padding_length  # or just skip if not used later

        # reindex
        reverse_index = {}
        for idx, x in enumerate(dfg):
            reverse_index[x[1]] = idx
        for idx, x in enumerate(dfg):
            dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)

        dfg_to_dfg = [x[-1] for x in dfg]
        dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
        length = len([tokenizer.cls_token])
        dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]

        # Comment -> tokens + ids
        nl_tokens = tokenizer.tokenize(raw_comment)[:args.nl_length - 2]
        nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]

        nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = args.nl_length - len(nl_ids)
        nl_ids += [tokenizer.pad_token_id] * padding_length
        nl_tokens += [tokenizer.pad_token] * padding_length  # or just skip if not used later

        # Concept alignment
        if compute_alignment:
            response = json.loads(obj["response"])
            concept_alignment, valid_code_spans, valid_comment_spans = build_concept_alignment(
                stepwise_descs=response["STEPWISE_DESCS"],
                comment_concepts=response["COMMENT_CONCEPTS"],
                alignment_map=response["ALIGNMENT_MAP"],
                nl_tokens=nl_tokens,
                code_tokens=code_tokens,
                tokenizer=tokenizer,
                example_idx=code_url
            )
            
            # Only print debug info for the first three examples
            if code_url in ['0', '1', '2']:
                print("\n" + "="*50)
                print(f"Processing example {code_url}")
                print("\n=== Input Data ===")
                print(f"Docstring: {obj['docstring']}")
                print(f"Raw Comment: {raw_comment}")
                print(f"Raw Code: {raw_code}")
                
                print("\n=== Stepwise Descriptions ===")
                for step in response["STEPWISE_DESCS"]:
                    step_name = list(step.keys())[0]
                    print(f"\n{step_name}:")
                    print(f"  Desc: {step[step_name]['desc']}")
                    print(f"  Code: {step[step_name]['code']}")
                
                print("\n=== Comment Concepts ===")
                for concept in response["COMMENT_CONCEPTS"]:
                    print(f"  {concept}")
                
                print("\n=== Alignment Map ===")
                for concept in response["ALIGNMENT_MAP"]:
                    print(f"  {concept}")
                
                print("\n=== Extracted Text Information ===")
                print(f"Comment Concepts: {valid_comment_spans}")
                print(f"Step Descriptions: {valid_code_spans}")
                
                print("\n=== Alignment Results ===")
                for i, alignment in enumerate(concept_alignment):
                    print(f"\nAlignment {i+1}:")
                    # Print actual text content
                    comment_spans = alignment[0]
                    code_spans = alignment[1]
                    
                    # Get comment text
                    comment_texts = []
                    for i in range(0, len(comment_spans), 2):
                        start, end = comment_spans[i], comment_spans[i+1]
                        comment_texts.append(" ".join(nl_tokens[start:end+1]))
                    
                    # Get code text
                    code_texts = []
                    for i in range(0, len(code_spans), 2):
                        start, end = code_spans[i], code_spans[i+1]
                        code_texts.append(" ".join(code_tokens[start:end+1]))
                    
                    print("  Comment spans:", comment_spans)
                    print("  Code spans:", code_spans)
                    print("  Comment text:", " | ".join(comment_texts))
                    print("  Code text:", " | ".join(code_texts))
                
                print("\n" + "="*50 + "\n")
                
                # Stop after processing the third example
                if code_url == '2':
                    print("\nReached third example, stopping for debug...")
                    import sys
                    sys.exit(0)
            
            return InputFeatures(
                code_tokens=code_tokens,
                code_ids=code_ids,
                position_idx=position_idx,
                dfg_to_code=dfg_to_code,
                dfg_to_dfg=dfg_to_dfg,
                nl_tokens=nl_tokens,
                nl_ids=nl_ids,
                code_url=code_url,
                ori2cur_pos=ori2cur_pos,
                concept_alignment=concept_alignment,
                valid_code_spans=valid_code_spans,
                valid_comment_spans=valid_comment_spans
            )
        else:
            return InputFeatures(code_tokens=code_tokens,
                                 code_ids=code_ids,
                                 position_idx=position_idx,
                                 dfg_to_code=dfg_to_code,
                                 dfg_to_dfg=dfg_to_dfg,
                                 nl_tokens=nl_tokens,
                                 nl_ids=nl_ids,
                                 code_url=code_url,
                                 ori2cur_pos=ori2cur_pos,
                                 )


def textdataset_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]], List[Tuple[str, List[int]]], List[Tuple[str, List[int]]]]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[List[List[int]]], List[List[Tuple[str, List[int]]]], List[List[Tuple[str, List[int]]]]]:

    code_ids, attn_masks, position_idxs, nl_ids, ori2cur_pos, alignments, valid_code_spans, valid_comment_spans = zip(*batch)

    return (
        torch.stack(code_ids),
        torch.stack(attn_masks),
        torch.stack(position_idxs),
        torch.stack(nl_ids),
        torch.stack(ori2cur_pos),
        alignments,  # keep as list of variable-length elements
        valid_code_spans,  # keep as list of variable-length elements
        valid_comment_spans
    )


def textdataset_noalign_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]], List[str], List[str]]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    code_ids, attn_masks, position_idxs, nl_ids, ori2cur_pos, alignments, valid_code_spans, valid_comment_spans = zip(*batch)

    return (
        torch.stack(code_ids),
        torch.stack(attn_masks),
        torch.stack(position_idxs),
        torch.stack(nl_ids),
        torch.stack(ori2cur_pos)
    )