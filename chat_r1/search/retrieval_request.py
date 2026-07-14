import requests
import time
import json

def passages2string(retrieval_result):
    format_reference = ''
    for idx, doc_item in enumerate(retrieval_result):
        format_reference += f"passage_id: {doc_item['document']['passage_id']} (Title: {doc_item['document']['title']}) {doc_item['document']['passage_text']}\n"

    return format_reference

url = "http://127.0.0.1:8002/retrieve"
payload = {
    "queries": ["What is the capital of France?", "Explain neural networks."],
    "topk": 3,
    "return_scores": True
}

RETRY_INTERVAL = 5
MAX_RETRIES = None

attempt = 0
while True:
    attempt += 1
    if MAX_RETRIES is not None and attempt > MAX_RETRIES:
        print(f"已达到最大重试次数 {MAX_RETRIES}，退出。")
        break

    try:
        print(f"正在尝试第 {attempt} 次调用 API...")
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        # ====== 新增：打印响应内容 ======
        print("✅ 成功获取响应，内容如下：")
        results = response.json()['result']
        results = [passages2string(result) for result in results]
        result = results[0].strip()
        # ================================

        # 成功完成后退出循环
        break

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        print(f"将在 {RETRY_INTERVAL} 秒后重试...")
        time.sleep(RETRY_INTERVAL)
        continue