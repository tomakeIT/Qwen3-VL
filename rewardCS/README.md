## Overview

`rewardCS` provides an **HTTP reward scoring service** based on a Qwen-VL model,  
which evaluates the delta progress between two time steps (image pairs) under the same task.

Components:
- `server.py`: starts a FastAPI service exposing `/init_task` and `/predict` endpoints  
- `client.py`: a simple Python client with command-line examples

For the most up-to-date usage patterns, please refer to the `main()` functions in `server.py` and `client.py`.

## Dependencies

### Server (`server.py`)

To run the FastAPI service and load the Qwen-VL model (tested with Python 3.10), install:

```bash
pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  transformers==4.57.1 \
  peft==0.17.1 \
  fastapi \
  uvicorn \
  pydantic \
  pillow

pip install "flash-attn==2.7.4.post1" --no-build-isolation
```

Notice: installing `flash-attn` may fail without a proxy/VPN.  
In that case, you can manually download a compatible wheel and install it locally.
