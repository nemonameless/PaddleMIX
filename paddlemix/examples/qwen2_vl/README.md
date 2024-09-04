# Qwen2-VL

## 1. 模型介绍

[Qwen2-VL](https://qwenlm.github.io/blog/qwen2-vl/) 是大规模视觉语言模型。可以以图像、文本、检测框、视频作为输入，并以文本和检测框作为输出。
本仓库提供paddle版本的Qwen2-VL-2B-Instruct和Qwen2-VL-7B-Instruct模型。


## 2 环境准备
- **python >= 3.10**
- tiktoken
> 注：tiktoken 要求python >= 3.8
- paddlepaddle-gpu >= 2.6.1
- paddlenlp >= 3.0.0

> 注：请确保安装了以上依赖，否则无法运行。同时，需要安装 paddlemix/external_ops 下的自定义OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置PYTHONPATH

## 3 快速开始
完成环境准备后，我们提供三种使用方式：

### a. 单轮预测
```bash
# qwen-vl
python paddlemix/examples/qwen_vl/run_predict.py \
--model_name_or_path "qwen-vl/qwen-vl-7b" \
--input_image "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg" \
--prompt "Generate the caption in English with grounding:" \
--dtype "bfloat16"
```
可配置参数说明：
  * `model_name_or_path`: 指定qwen_vl系列的模型名字或权重路径，默认 qwen-vl/qwen-vl-7b
  * `seed` :指定随机种子，默认1234。
  * `visual:` :设置是否可视化结果，默认True。
  * `output_dir` :指定可视化图片保存路径。
  * `dtype` :设置数据类型，默认bfloat16,支持float32、bfloat16、float16。
  * `input_image` :输入图片路径或url，默认None。
  * `prompt` :输入prompt。

### b. 多轮对话
```bash
python paddlemix/examples/qwen_vl/chat_demo.py
```

### c. 


## 4 模型微调

