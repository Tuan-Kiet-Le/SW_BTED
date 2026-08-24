# Qwen3 tokenizer/input-length audit status

The successful Kaggle version 10 run records `max_length = 2048`, CUDA execution, and the model identifier `Qwen/Qwen3-Embedding-4B`. It uses `Qwen2Tokenizer`, whose declared tokenizer maximum is 131072, but the configured encoding cutoff is 2048.

The direct audit found 178 documents, with 18/178 (`10.11%`) over the configured cutoff; median length was 637 tokens, P95 was 2504.2, and maximum was 4156 tokens.

Therefore Qwen3's reported result is a 2048-token truncation protocol, not a truncation-free result. The artifact is stored under `kaggle/qwen3_results_v10/qwen3_results/`.
