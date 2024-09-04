
import paddle
from matplotlib.pyplot import text
from paddlemix.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from paddlemix.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from paddlemix.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
#from paddlenlp.transformers import AutoProcessor, AutoTokenizer
from paddlenlp.transformers import Qwen2Tokenizer
from paddlemix.models.qwen2_vl.vision_process import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen2-VL-2B-Instruct_pd", dtype="bfloat16")
#model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen2-VL-2B-Instruct_pd")

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = Qwen2VLProcessor.from_pretrained("Qwen2-VL-2B-Instruct_pd")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

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

# Preparation for inference
image_inputs, video_inputs = process_vision_info(messages)

#from paddlenlp.transformers.tokenizer_utils import ChatTemplateMixin

# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
tokenizer = Qwen2Tokenizer.from_pretrained("Qwen2-VL-2B-Instruct_pd")

# # TODO:
# tokenizer.added_tokens_encoder =  {'<|endoftext|>': 151643, '<|im_start|>': 151644, '<|im_end|>': 151645, '<img>': 151646, '</img>': 151647, '<IMG_CONTEXT>': 151648, '<quad>': 151649, '</quad>': 151650, '<ref>': 151651, '</ref>': 151652, '<box>': 151653, '</box>': 151654}
# tokenizer.added_tokens_decoder = {v: k for k, v in tokenizer.added_tokens_encoder.items()}


# text = tokenizer.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
text = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pd",
)


# import numpy as np
# inputs["input_ids"] = paddle.to_tensor(np.load('inputs/input_ids.npy').astype(np.int64))
# inputs["attention_mask"] = paddle.to_tensor(np.load('inputs/attention_mask.npy').astype(np.int64))
# inputs["pixel_values"] = paddle.to_tensor(np.load('inputs/pixel_values.npy').astype(np.float32))
# inputs["image_grid_thw"] = paddle.to_tensor(np.load('inputs/image_grid_thw.npy').astype(np.int64))



# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
