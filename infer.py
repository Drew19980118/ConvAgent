import transformers
import torch
import random
import re
from datasets import load_dataset
import requests

context = """User: what was australia's contribution to the battle of normandy
Assistant: the army personnel and thousands of australian airmen took part in the battle.
User: was the battle fought in australia
Assistant: unanswerable
User: when was the battle fought
Assistant: 1944
User: who fought in this battle
Assistant: australians and british
User: was this battle part of a bigger war
Assistant: unanswerable"""

question = "User: were there any preparations made before the invasion?"

# Model ID and device setup
model_id = "slupart/ChatR1-topiocqa-qwen2.5-3b-it-ppo"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

question = question.strip()
if question[-1] != '?':
    question += '?'
curr_eos = [151645, 151643]  # for Qwen2.5 series models
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

# Prepare the message
prompt = f"""You are a helpful assistant tasked with answering a user query. \
Your primary goal is to generate a complete and informative answer.\n\n \
If the query is ambiguous or refers to earlier context (e.g., pronouns or ellipsis), use the conversation history provided below to resolve it.\n\n \
- Always perform your reasoning inside <think>...</think>.\n- If external information is needed, use <search>your query</search>. \n \
- Retrieved documents will appear between <information>...</information>.\n \
- You may issue multiple <search> queries if needed.\n \
- Once you have enough information, provide a complete answer within <answer>...</answer>.\n\n \
Conversation context:\n<context>\n{context}\n</context>\n\n \
User query:\n{question}\n"""

# Initialize the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")


# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [tokenizer.encode(t, add_special_tokens=False) for t in target_sequences]
        self.target_lengths = [len(t) for t in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        targets = [torch.as_tensor(t, device=input_ids.device) for t in self.target_ids]
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True
        return False


def get_query(text):
    # Only look at content after the last <|im_start|>assistant marker
    # so earlier turns / instructions can't interfere.
    marker = "<|im_start|>assistant"
    idx = text.rfind(marker)
    if idx != -1:
        text = text[idx + len(marker):]
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1] if matches else None


def get_answer(text):
    """Return the last non-empty <answer>...</answer> content, or None."""
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = pattern.findall(text)
    for m in reversed(matches):
        if m.strip():
            return m.strip()
    return None


def search(query: str):
    payload = {"queries": [query], "topk": 3, "return_scores": True}
    results = requests.post("http://127.0.0.1:8002/retrieve", json=payload).json()['result']

    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


# Initialize the stopping criteria
target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

cnt = 0
MAX_TURNS = 10  # safety cap in case something else goes wrong

if tokenizer.chat_template:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )

print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
print(prompt)

final_answer = None

while True:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1024,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
    )

    generated_tokens = outputs[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # --- NEW: break if a non-empty <answer> was produced in this step ---
    answer_in_step = get_answer(output_text)
    if answer_in_step:
        print(output_text)
        print("\n[Stopping: non-empty <answer> detected in generated step]")
        final_answer = answer_in_step
        break
    # -------------------------------------------------------------------

    # Natural EOS stop
    if outputs[0][-1].item() in curr_eos:
        print(output_text)
        # Still try to extract an answer from the full text
        final_answer = get_answer(output_text)
        break

    tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=False))
    if tmp_query:
        print(f"\n[Search query detected: '{tmp_query}']")
    search_results = search(tmp_query) if tmp_query else ''

    search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
    prompt += search_text
    cnt += 1
    print(search_text)

    # Safety cap
    if cnt >= MAX_TURNS:
        print(f"\n[Stopping: reached MAX_TURNS={MAX_TURNS}]")
        final_answer = get_answer(prompt)
        break

print("\n\n################# [Final Answer] ##################")
print(final_answer if final_answer else "[No answer extracted]")