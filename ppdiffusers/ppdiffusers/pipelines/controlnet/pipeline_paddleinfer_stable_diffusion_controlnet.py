# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from ..stable_diffusion.pipeline_paddleinfer_stable_diffusion import (
    PaddleInferStableDiffusionPipeline,
)


class PaddleInferStableDiffusionControlNetPipeline(PaddleInferStableDiffusionPipeline):
    def __call__(
        self,
        *args,
        **kwargs,
    ):
        controlnet_cond = kwargs.pop("controlnet_cond", None)
        image = kwargs.pop("image", None)
        if controlnet_cond is None:
            kwargs["controlnet_cond"] = image
        else:
            kwargs["controlnet_cond"] = controlnet_cond
        return super().__call__(*args, **kwargs)
