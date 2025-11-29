## 简介

`rewardCS` 提供一个 **HTTP 奖励打分服务**，基于 Qwen-VL 模型评估同一任务下两帧图像的进度差（delta progress）。  
包含：
- `server.py`：启动 FastAPI 服务，对外暴露 `/init_task` 和 `/predict` 接口  
- `client.py`：简单的 Python 客户端与命令行示例

具体使用看sever.py client.py 的main()函数

## 依赖
### 服务端（`server.py`）依赖

运行 FastAPI 服务以及加载 Qwen-VL 模型需要这些库, 我们使用的是python 3.10：

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
Notice: 这里的flash-attn要是没有代理或者VPN的话就会安装失败 可以手动下载Wheel包安装
