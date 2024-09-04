# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import torch
from matplotlib.pyplot import text
from paddlenlp.transformers import Qwen2Tokenizer

from paddlemix.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from paddlemix.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLConfig,
)
from paddlemix.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor


from paddlemix.models.qwen2_vl.vision_process import process_vision_info


from transformers import Qwen2Tokenizer as Qwen2Tokenizer_HF
from transformers import Qwen2VLConfig as Qwen2VLConfig_HF
from transformers import (
    Qwen2VLForConditionalGeneration as Qwen2VLForConditionalGeneration_HF,
)
from transformers import Qwen2VLProcessor as Qwen2VLProcessor_HF

paddle_ckpt_path = "Qwen2-VL-2B-Instruct_pd"
torch_ckpt_path = "Qwen2-VL-2B-Instruct"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

image_inputs, video_inputs = process_vision_info(messages)

text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n"
tokenizer = Qwen2Tokenizer.from_pretrained(paddle_ckpt_path)
tokenizer_torch = Qwen2Tokenizer_HF.from_pretrained(torch_ckpt_path)

processor_paddle = Qwen2VLProcessor.from_pretrained(paddle_ckpt_path)
processor_torch = Qwen2VLProcessor_HF.from_pretrained(torch_ckpt_path)

inputs_paddle = processor_paddle(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pd",
)

inputs_torch = processor_torch(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

config_paddle = Qwen2VLConfig.from_pretrained(paddle_ckpt_path)
# config_paddle.num_hidden_layers = 2
# config_paddle.depth = 2
model_paddle = Qwen2VLForConditionalGeneration.from_pretrained(paddle_ckpt_path, config=config_paddle, dtype="float16")


config_torch = Qwen2VLConfig_HF.from_pretrained(torch_ckpt_path)
config_torch.torch_dtype = torch.float16
config_torch.dtype = "float16"
# config_torch.num_hidden_layers = 2
# config_torch.depth = 2
#config_torch._attn_implementation = "eager"
model_torch = Qwen2VLForConditionalGeneration_HF.from_pretrained(torch_ckpt_path, config=config_torch) #.cuda()



# # Inference: paddle
# generated_ids = model_paddle.generate(**inputs_paddle, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_paddle.input_ids, generated_ids)
# ]
# output_text_paddle = processor_paddle.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text_paddle)



# Inference: torch
generated_ids = model_torch.generate(**inputs_torch, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_torch.input_ids, generated_ids)
]
output_text_torch = processor_torch.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text_torch)



# # padiff 3.0 develop
# from padiff import auto_diff
# input_torch = {
#     "input_ids": torch.LongTensor(inputs_paddle["input_ids"].numpy()), #.cuda(),
#     "attention_mask": torch.LongTensor(inputs_paddle["attention_mask"].numpy()), #.cuda(),
#     "pixel_values": torch.FloatTensor(inputs_paddle["pixel_values"].numpy()), #.cuda(),
#     "image_grid_thw": torch.LongTensor(inputs_paddle["image_grid_thw"].numpy()), #.cuda(),
# }

# input_paddle = {
#     "input_ids": inputs_paddle["input_ids"],
#     "attention_mask": inputs_paddle["attention_mask"],
#     "pixel_values": inputs_paddle["pixel_values"],
#     "image_grid_thw": inputs_paddle["image_grid_thw"]
# }


# # output_paddle = model_paddle(**input_paddle)[0]

# # inp = ({"input_ids": torch.LongTensor(inputs_paddle["input_ids"].numpy())}, {"input_ids": inputs_paddle["input_ids"]})
# inp = (input_torch, input_paddle)
# auto_diff(model_torch, model_paddle, inp, atol=1e-3, auto_init=False, diff_phase="forward", compare_mode="strict")
