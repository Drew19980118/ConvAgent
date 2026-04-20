
index_file=collection/topiocqa/index/e5_Flat.index
corpus_file=collection/topiocqa.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python chat_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu
