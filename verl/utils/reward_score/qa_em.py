# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


# def extract_solution(solution_str):
#     """Extract the equation from the solution string."""
#     # Remove everything before the first "Assistant:"
#     # if "Assistant:" in solution_str:
#     #     solution_str = solution_str.split("Assistant:", 1)[1]
#     # elif "<|im_start|>assistant" in solution_str:
#     #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
#     # else:
#     #     return None
#     # solution_str = solution_str.split('\n')[-1]

#     answer_pattern = r'<answer>(.*?)</answer>'
#     match = re.finditer(answer_pattern, solution_str, re.DOTALL)
#     matches = list(match)
    
#     # If there are 0 or exactly 1 matches, return None
#     if len(matches) <= 1:
#         return None
    
#     # If there are 2 or more matches, return the last one
#     return matches[-1].group(1).strip()


def extract_solution(solution_str):
    """Extract the last non-empty <answer>...</answer> content, or return None."""
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    if not matches:
        return None

    # Get the last match
    last_content = matches[-1].group(1).strip()

    if last_content.strip().lower() == "and":
        # If the last content is just "and", return None
        return None

    return last_content if last_content else None


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score

def f1_score(prediction, ground_truth):
    """Compute the token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    common = set(pred_tokens) & set(gold_tokens)
    num_same = len(common)
    
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return int(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_score_f1(solution_str, ground_truth):
    """The scoring function based on F1 score.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format (applied if no answer found)
        score: the maximum score (scales the F1 score)
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if isinstance(ground_truth['target'], str):
            targets = [ground_truth['target']]
        else:
            targets = ground_truth['target']

        f1_scores = [f1_score(answer, target) for target in targets]
        best_f1 = max(f1_scores)
        return best_f1



# from me

# def extract_prompt_output(full_output, prompt_hint="User query:"):
#     """
#     Extracts model output portion by removing the initial prompt template,
#     assuming prompt ends with a known string like 'User query: ...'
#     """
#     # Find the last occurrence of 'User query:' which marks the end of the prompt
#     last_prompt_pos = full_output.rfind(prompt_hint)
#     if last_prompt_pos != -1:
#         # Return everything after the prompt (assumes model output starts after it)
#         return full_output[last_prompt_pos + len(prompt_hint):].strip()
#     return full_output.strip()  # fallback: use full string


# def format_reward(solution_str):
#     """
#     Reward if the output includes one <think>...</think> and one <answer>...</answer> block,
#     excluding prompt parts.
#     """
#     # Remove prompt contamination
#     model_output = extract_prompt_output(solution_str)

#     think_blocks = re.findall(r'<think>.*?</think>', model_output, re.DOTALL)
#     answer_blocks = re.findall(r'<answer>.*?</answer>', model_output, re.DOTALL)

#     if len(think_blocks) == 1 and len(answer_blocks) == 1:
#         return 1.0
#     elif len(answer_blocks) > 0:
#         return 0.5
#     else:
#         return 0.0

# def compute_f1_format_reward(solution_str, ground_truth, f1_weight=1.0, format_weight=0.3):
#     f1 = compute_score_f1(solution_str, ground_truth)
#     fmt = format_reward(solution_str)
#     return f1_weight * f1 + format_weight * fmt








# from search r1 v2 paper

# def is_valid_sequence(text):
#     # Find the position of "<|im_start|>assistant" with potential whitespace
#     assistant_pattern = r"<\|im_start\|>assistant\s*"
#     assistant_match = re.search(assistant_pattern, text)
    
#     if not assistant_match:
#         return False, "Missing assistant marker"
    
#     # Extract the content after the assistant marker
#     start_pos = assistant_match.end()
#     content = text[start_pos:]
    
#     # Check for balanced tags
#     tags_to_check = ["think", "search", "information", "answer"]
#     for tag in tags_to_check:
#         opening_count = len(re.findall(f"<{tag}>", content))
#         closing_count = len(re.findall(f"</{tag}>", content))
#         if opening_count != closing_count:
#             return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
#     # Now check for proper sequence pattern and no extraneous content
    
#     # 1. First split the content by any tags we recognize
#     split_pattern = r"(</?(?:think|search|information|answer)>)"
#     parts = re.split(split_pattern, content)
    
#     # 2. Keep track of the current position in the expected sequence
#     state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end
    
#     # 3. Check each part
#     for i, part in enumerate(parts):
#         # Skip empty parts
#         if not part.strip():
#             continue
            
#         # Check if this is a tag
#         if re.match(r"</?(?:think|search|information|answer)>", part):
#             # This is a tag, check if it's valid in the current state
#             if part == "<think>" and state in ["start", "information"]:
#                 state = "in_think"
#             elif part == "</think>" and state == "in_think":
#                 state = "after_think"
#             elif part == "<search>" and state == "after_think":
#                 state = "in_search"
#             elif part == "</search>" and state == "in_search":
#                 state = "after_search"
#             elif part == "<information>" and state == "after_search":
#                 state = "in_information"
#             elif part == "</information>" and state == "in_information":
#                 state = "information"
#             elif part == "<answer>" and state == "after_think":
#                 state = "in_answer"
#             elif part == "</answer>" and state == "in_answer":
#                 state = "end"
#             else:
#                 return False, f"Unexpected tag {part} in state {state}"
#         else:
#             # This is content, check if it's valid in the current state
#             if state in ["in_think", "in_search", "in_information", "in_answer"]:
#                 # Content is allowed inside tags
#                 pass
#             elif state in ["start", "after_think", "after_search", "information"]:
#                 # Only whitespace is allowed between tags
#                 if part.strip():
#                     return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
#             else:
#                 return False, f"Unexpected content in state {state}"
    
#     # Check final state
#     if state != "end":
#         return False, f"Incomplete sequence, ended in state {state}"
        
#     return True, "Valid sequence format"


# # format from search r1 v2
# def compute_score_f1(solution_str, ground_truth, method='strict', structure_format_score=0.1, final_format_score=0, retrieval_score=0, format_score=0, score=1.):
#     """The scoring function for exact match (EM).

#     Args:
#         solution_str: the solution text
#         ground_truth: the ground truth
#         method: the method to extract the solution, choices are 'strict' and 'flexible'
#         format_score: the score for the format
#         score: the score for the correct answer
#     """
#     is_valid_format, _ = is_valid_sequence(solution_str)
#     # retrieval_correct = False
#     # if is_valid_format:
#     #     retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
#     answer = extract_solution(solution_str=solution_str)
#     do_print = random.randint(1, 64) == 1
    
#     if do_print:
#         print(f"--------------------------------")
#         print(f"Golden answers: {ground_truth['target']}")
#         print(f"Extracted answer: {answer}")
#         print(f"Solution string: {solution_str}")
            
#     if answer is None:
#         if is_valid_format:
#             return structure_format_score # +0.1
#         else:
#             return 0
#     else:
#         if isinstance(ground_truth['target'], str):
#             targets = [ground_truth['target']]
#         else:
#             targets = ground_truth['target']

#         f1_scores = [f1_score(answer, target) for target in targets]
#         score = max(f1_scores)
#         # if score:
#         if is_valid_format:
#             return score + structure_format_score # +0.1
#         else:
#             return score - structure_format_score # -0.1



