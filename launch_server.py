import os

from sglang.test.test_utils import is_in_ci
from sglang.utils import wait_for_server, print_highlight, terminate_process

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

# This is equivalent to running the following command in your terminal

# python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0

os.environ["CUDA_VISIBLE_DEVICES"] = "0,5"

server_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server             \
    --model-path neuralmagic/DeepSeek-Coder-V2-Instruct-FP8  \
    --trust-remote-code                     \
    --disable-radix-cache                   \
    --quantization fp8                    \
    --kv-cache-dtype fp8_e5m2             \
    --mem-fraction-static 0.7              \
    --max-running-requests 4096             \
    --max-prefill-tokens  16384             \
    --chunked-prefill-size 16384            \
    --disable-cuda-graph \
    --dp 1 --enable-dp-attention            \
    --tp 2 \
    --enable-ep-moe
"""
)

# server_process, port = launch_server_cmd(
#     """
# python3 -m sglang.launch_server             \
#     --model-path allenai/OLMoE-1B-7B-0924 \
#     --trust-remote-code                     \
#     --disable-radix-cache                   \
#     --quantization fp8                    \
#     --kv-cache-dtype fp8_e5m2             \
#     --mem-fraction-static 0.7              \
#     --max-running-requests 4096             \
#     --max-prefill-tokens  16384             \
#     --chunked-prefill-size 16384            \
#     --disable-cuda-graph \
#     --dp 2 --enable-dp-attention            \
# """
# )

wait_for_server(f"http://localhost:{port}")
print(f"Server started on http://localhost:{port}")
