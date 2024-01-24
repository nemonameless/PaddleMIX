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

from typing import TYPE_CHECKING

from .patches import *
from .utils import (
    PPDIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_fastdeploy_available,
    is_k_diffusion_available,
    is_librosa_available,
    is_note_seq_available,
    is_paddle_available,
    is_paddlenlp_available,
    is_paddlesde_available,
    is_scipy_available,
)
from .version import VERSION as __version__

# Lazy Import based on
# https://github.com/huggingface/transformers/blob/main/src/transformers/__init__.py

# When adding a new object to this init, please add it to `_import_structure`. The `_import_structure` is a dictionary submodule to list of object names,
# and is used to defer the actual importing for when the objects are requested.
# This way `import diffusers` provides the names in the namespace without actually importing anything (and especially none of the backends).

_import_structure = {
    "configuration_utils": ["ConfigMixin"],
    "models": [],
    "pipelines": [],
    "schedulers": [],
    "utils": [
        "OptionalDependencyNotAvailable",
        "is_inflect_available",
        "is_pp_invisible_watermark_available",
        "is_k_diffusion_available",
        "is_k_diffusion_version",
        "is_librosa_available",
        "is_note_seq_available",
        "is_onnx_available",
        "is_fastdeploy_available",
        "is_scipy_available",
        "is_paddle_available",
        "is_paddle_version",
        "is_paddlesde_available",
        "is_paddlenlp_available",
        "is_paddlenlp_version",
        "is_unidecode_available",
        # NEW ADD
        "is_einops_available",
        "is_torch_available",
        "is_transformers_available",
        "logging",
    ],
}

try:
    if not is_fastdeploy_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_fastdeploy_objects  # noqa F403

    _import_structure["utils.dummy_fastdeploy_objects"] = [
        name for name in dir(dummy_fastdeploy_objects) if not name.startswith("_")
    ]

else:
    _import_structure["pipelines"].extend(
        ["FastDeployRuntimeModel", "FastDeployDiffusionPipelineMixin", "FastDeployDiffusionXLPipelineMixin"]
    )

try:
    if not is_paddle_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_paddle_objects  # noqa F403

    _import_structure["utils.dummy_paddle_objects"] = [
        name for name in dir(dummy_paddle_objects) if not name.startswith("_")
    ]

else:
    _import_structure["models"].extend(
        [
            "AsymmetricAutoencoderKL",
            "AutoencoderKL",
            "AutoencoderKLTemporalDecoder",
            "AutoencoderTiny",
            "ConsistencyDecoderVAE",
            "ControlNetModel",
            "Kandinsky3UNet",
            "ModelMixin",
            "MotionAdapter",
            "MultiAdapter",
            "PriorTransformer",
            "T2IAdapter",
            "T5FilmDecoder",
            "Transformer2DModel",
            "UNet1DModel",
            "UNet2DConditionModel",
            "UNet2DModel",
            "UNet3DConditionModel",
            "UNetMotionModel",
            "UNetSpatioTemporalConditionModel",
            "VQModel",
        ]
    )

    _import_structure["optimization"] = [
        "get_constant_schedule",
        "get_constant_schedule_with_warmup",
        "get_cosine_schedule_with_warmup",
        "get_cosine_with_hard_restarts_schedule_with_warmup",
        "get_linear_schedule_with_warmup",
        "get_polynomial_decay_schedule_with_warmup",
        "get_scheduler",
    ]
    _import_structure["pipelines"].extend(
        [
            "AudioPipelineOutput",
            "AutoPipelineForImage2Image",
            "AutoPipelineForInpainting",
            "AutoPipelineForText2Image",
            "ConsistencyModelPipeline",
            "DanceDiffusionPipeline",
            "DDIMPipeline",
            "DDPMPipeline",
            "DiffusionPipeline",
            "DiTPipeline",
            "ImagePipelineOutput",
            "KarrasVePipeline",
            "LDMPipeline",
            "LDMSuperResolutionPipeline",
            "PNDMPipeline",
            "RePaintPipeline",
            "ScoreSdeVePipeline",
        ]
    )
    _import_structure["schedulers"].extend(
        [
            "CMStochasticIterativeScheduler",
            "DDIMInverseScheduler",
            "DDIMParallelScheduler",
            "DDIMScheduler",
            "DDPMParallelScheduler",
            "DDPMScheduler",
            "DDPMWuerstchenScheduler",
            "DEISMultistepScheduler",
            "DPMSolverMultistepInverseScheduler",
            "DPMSolverMultistepScheduler",
            "DPMSolverSinglestepScheduler",
            "EulerAncestralDiscreteScheduler",
            "EulerDiscreteScheduler",
            "HeunDiscreteScheduler",
            "IPNDMScheduler",
            "KarrasVeScheduler",
            "KDPM2AncestralDiscreteScheduler",
            "KDPM2DiscreteScheduler",
            "LCMScheduler",
            "PNDMScheduler",
            "RePaintScheduler",
            "SchedulerMixin",
            "ScoreSdeVeScheduler",
            "UnCLIPScheduler",
            "UniPCMultistepScheduler",
            "VQDiffusionScheduler",
        ]
    )
    _import_structure["training_utils"] = ["EMAModel"]

try:
    if not (is_paddle_available() and is_scipy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_paddle_and_scipy_objects  # noqa F403

    _import_structure["utils.dummy_paddle_and_scipy_objects"] = [
        name for name in dir(dummy_paddle_and_scipy_objects) if not name.startswith("_")
    ]

else:
    _import_structure["schedulers"].extend(["LMSDiscreteScheduler"])

try:
    if not (is_paddle_available() and is_paddlesde_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_paddle_and_paddlesde_objects  # noqa F403

    _import_structure["utils.dummy_paddle_and_paddlesde_objects"] = [
        name for name in dir(dummy_paddle_and_paddlesde_objects) if not name.startswith("_")
    ]

else:
    _import_structure["schedulers"].extend(["DPMSolverSDEScheduler"])

try:
    if not (is_paddle_available() and is_paddlenlp_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_paddle_and_paddlenlp_objects  # noqa F403

    _import_structure["utils.dummy_paddle_and_paddlenlp_objects"] = [
        name for name in dir(dummy_paddle_and_paddlenlp_objects) if not name.startswith("_")
    ]

else:
    _import_structure["pipelines"].extend(
        [
            "AltDiffusionImg2ImgPipeline",
            "AltDiffusionPipeline",
            "AnimateDiffPipeline",
            "AudioLDM2Pipeline",
            "AudioLDM2ProjectionModel",
            "AudioLDM2UNet2DConditionModel",
            "AudioLDMPipeline",
            "BlipDiffusionControlNetPipeline",
            "BlipDiffusionPipeline",
            "CLIPImageProjection",
            "CycleDiffusionPipeline",
            "IFImg2ImgPipeline",
            "IFImg2ImgSuperResolutionPipeline",
            "IFInpaintingPipeline",
            "IFInpaintingSuperResolutionPipeline",
            "IFPipeline",
            "IFSuperResolutionPipeline",
            "ImageTextPipelineOutput",
            "Kandinsky3Img2ImgPipeline",
            "Kandinsky3Pipeline",
            "KandinskyCombinedPipeline",
            "KandinskyImg2ImgCombinedPipeline",
            "KandinskyImg2ImgPipeline",
            "KandinskyInpaintCombinedPipeline",
            "KandinskyInpaintPipeline",
            "KandinskyPipeline",
            "KandinskyPriorPipeline",
            "KandinskyV22CombinedPipeline",
            "KandinskyV22ControlnetImg2ImgPipeline",
            "KandinskyV22ControlnetPipeline",
            "KandinskyV22Img2ImgCombinedPipeline",
            "KandinskyV22Img2ImgPipeline",
            "KandinskyV22InpaintCombinedPipeline",
            "KandinskyV22InpaintPipeline",
            "KandinskyV22Pipeline",
            "KandinskyV22PriorEmb2EmbPipeline",
            "KandinskyV22PriorPipeline",
            "LatentConsistencyModelImg2ImgPipeline",
            "LatentConsistencyModelPipeline",
            "LDMTextToImagePipeline",
            "MusicLDMPipeline",
            "PaintByExamplePipeline",
            "PixArtAlphaPipeline",
            "SemanticStableDiffusionPipeline",
            "ShapEImg2ImgPipeline",
            "ShapEPipeline",
            "StableDiffusionAdapterPipeline",
            "StableDiffusionAttendAndExcitePipeline",
            "StableDiffusionControlNetImg2ImgPipeline",
            "StableDiffusionControlNetInpaintPipeline",
            "StableDiffusionControlNetPipeline",
            "StableDiffusionDepth2ImgPipeline",
            "StableDiffusionDiffEditPipeline",
            "StableDiffusionGLIGENPipeline",
            "StableDiffusionGLIGENTextImagePipeline",
            "StableDiffusionImageVariationPipeline",
            "StableDiffusionImg2ImgPipeline",
            "StableDiffusionInpaintPipeline",
            "StableDiffusionInpaintPipelineLegacy",
            "StableDiffusionInstructPix2PixPipeline",
            "StableDiffusionLatentUpscalePipeline",
            "StableDiffusionLDM3DPipeline",
            "StableDiffusionModelEditingPipeline",
            "StableDiffusionPanoramaPipeline",
            "StableDiffusionParadigmsPipeline",
            "StableDiffusionPipeline",
            "StableDiffusionPipelineSafe",
            "StableDiffusionPix2PixZeroPipeline",
            "StableDiffusionSAGPipeline",
            "StableDiffusionUpscalePipeline",
            "StableDiffusionXLAdapterPipeline",
            "StableDiffusionXLControlNetImg2ImgPipeline",
            "StableDiffusionXLControlNetInpaintPipeline",
            "StableDiffusionXLControlNetPipeline",
            "StableDiffusionXLImg2ImgPipeline",
            "StableDiffusionXLInpaintPipeline",
            "StableDiffusionXLInstructPix2PixPipeline",
            "StableDiffusionXLPipeline",
            "StableUnCLIPImg2ImgPipeline",
            "StableUnCLIPPipeline",
            "StableVideoDiffusionPipeline",
            "TextToVideoSDPipeline",
            "TextToVideoZeroPipeline",
            "TextToVideoZeroSDXLPipeline",
            "UnCLIPImageVariationPipeline",
            "UnCLIPPipeline",
            "UniDiffuserModel",
            "UniDiffuserPipeline",
            "UniDiffuserTextDecoder",
            "VersatileDiffusionDualGuidedPipeline",
            "VersatileDiffusionImageVariationPipeline",
            "VersatileDiffusionPipeline",
            "VersatileDiffusionTextToImagePipeline",
            "VideoToVideoSDPipeline",
            "VQDiffusionPipeline",
            "WuerstchenCombinedPipeline",
            "WuerstchenDecoderPipeline",
            "WuerstchenPriorPipeline",
        ]
    )

try:
    if not (is_paddle_available() and is_paddlenlp_available() and is_k_diffusion_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_paddle_and_paddlenlp_and_k_diffusion_objects  # noqa F403

    _import_structure["utils.dummy_paddle_and_paddlenlp_and_k_diffusion_objects"] = [
        name for name in dir(dummy_paddle_and_paddlenlp_and_k_diffusion_objects) if not name.startswith("_")
    ]

else:
    _import_structure["pipelines"].extend(["StableDiffusionKDiffusionPipeline"])

try:
    if not (is_paddle_available() and is_paddlenlp_available() and is_fastdeploy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_paddle_and_paddlenlp_and_fastdeploy_objects  # noqa F403

    _import_structure["utils.dummy_paddle_and_paddlenlp_and_fastdeploy_objects"] = [
        name for name in dir(dummy_paddle_and_paddlenlp_and_fastdeploy_objects) if not name.startswith("_")
    ]

else:
    _import_structure["pipelines"].extend(
        [
            "FastDeployStableDiffusionImg2ImgPipeline",
            "FastDeployStableDiffusionInpaintPipeline",
            "FastDeployStableDiffusionInpaintPipelineLegacy",
            "FastDeployStableDiffusionPipeline",
            "FastDeployStableDiffusionMegaPipeline",
            "FastDeployCycleDiffusionPipeline",
            "FastDeployStableDiffusionControlNetPipeline",
            "FastDeployStableDiffusionUpscalePipeline",
        ]
    )

try:
    if not (is_paddle_available() and is_librosa_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_paddle_and_librosa_objects  # noqa F403

    _import_structure["utils.dummy_paddle_and_librosa_objects"] = [
        name for name in dir(dummy_paddle_and_librosa_objects) if not name.startswith("_")
    ]

else:
    _import_structure["pipelines"].extend(["AudioDiffusionPipeline", "Mel"])

try:
    if not (is_paddlenlp_available() and is_paddle_available() and is_note_seq_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_paddle_and_paddlenlp_and_note_seq_objects  # noqa F403

    _import_structure["utils.dummy_paddle_and_paddlenlp_and_note_seq_objects"] = [
        name for name in dir(dummy_paddle_and_paddlenlp_and_note_seq_objects) if not name.startswith("_")
    ]


else:
    _import_structure["pipelines"].extend(["SpectrogramDiffusionPipeline"])

try:
    if not (is_note_seq_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_note_seq_objects  # noqa F403

    _import_structure["utils.dummy_note_seq_objects"] = [
        name for name in dir(dummy_note_seq_objects) if not name.startswith("_")
    ]


else:
    _import_structure["pipelines"].extend(["MidiProcessor"])

if TYPE_CHECKING or PPDIFFUSERS_SLOW_IMPORT:
    from .configuration_utils import ConfigMixin

    try:
        if not is_fastdeploy_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_fastdeploy_objects import *  # noqa F403
    else:
        from .pipelines import (
            FastDeployDiffusionPipelineMixin,
            FastDeployDiffusionXLPipelineMixin,
            FastDeployRuntimeModel,
        )

    try:
        if not is_paddle_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_paddle_objects import *  # noqa F403
    else:
        from .models import (
            AsymmetricAutoencoderKL,
            AutoencoderKL,
            AutoencoderKLTemporalDecoder,
            AutoencoderTiny,
            ConsistencyDecoderVAE,
            ControlNetModel,
            Kandinsky3UNet,
            ModelMixin,
            MotionAdapter,
            MultiAdapter,
            PriorTransformer,
            T2IAdapter,
            T5FilmDecoder,
            Transformer2DModel,
            UNet1DModel,
            UNet2DConditionModel,
            UNet2DModel,
            UNet3DConditionModel,
            UNetMotionModel,
            UNetSpatioTemporalConditionModel,
            VQModel,
        )
        from .optimization import (
            get_constant_schedule,
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
            get_cosine_with_hard_restarts_schedule_with_warmup,
            get_linear_schedule_with_warmup,
            get_polynomial_decay_schedule_with_warmup,
            get_scheduler,
        )
        from .pipelines import (
            AudioPipelineOutput,
            AutoPipelineForImage2Image,
            AutoPipelineForInpainting,
            AutoPipelineForText2Image,
            BlipDiffusionControlNetPipeline,
            BlipDiffusionPipeline,
            ConsistencyModelPipeline,
            DanceDiffusionPipeline,
            DDIMPipeline,
            DDPMPipeline,
            DiffusionPipeline,
            DiTPipeline,
            ImagePipelineOutput,
            KarrasVePipeline,
            LDMPipeline,
            LDMSuperResolutionPipeline,
            PNDMPipeline,
            RePaintPipeline,
            ScoreSdeVePipeline,
        )
        from .schedulers import (
            CMStochasticIterativeScheduler,
            DDIMInverseScheduler,
            DDIMParallelScheduler,
            DDIMScheduler,
            DDPMParallelScheduler,
            DDPMScheduler,
            DDPMWuerstchenScheduler,
            DEISMultistepScheduler,
            DPMSolverMultistepInverseScheduler,
            DPMSolverMultistepScheduler,
            DPMSolverSinglestepScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            HeunDiscreteScheduler,
            IPNDMScheduler,
            KarrasVeScheduler,
            KDPM2AncestralDiscreteScheduler,
            KDPM2DiscreteScheduler,
            LCMScheduler,
            PNDMScheduler,
            RePaintScheduler,
            SchedulerMixin,
            ScoreSdeVeScheduler,
            UnCLIPScheduler,
            UniPCMultistepScheduler,
            VQDiffusionScheduler,
        )
        from .training_utils import EMAModel

    try:
        if not (is_paddle_available() and is_scipy_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_paddle_and_scipy_objects import *  # noqa F403
    else:
        from .schedulers import LMSDiscreteScheduler

    try:
        if not (is_paddle_available() and is_paddlesde_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_paddle_and_paddlesde_objects import *  # noqa F403
    else:
        from .schedulers import DPMSolverSDEScheduler

    try:
        if not (is_paddle_available() and is_paddlenlp_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_paddle_and_paddlenlp_objects import *  # noqa F403
    else:
        from .pipelines import (
            AltDiffusionImg2ImgPipeline,
            AltDiffusionPipeline,
            AnimateDiffPipeline,
            AudioLDM2Pipeline,
            AudioLDM2ProjectionModel,
            AudioLDM2UNet2DConditionModel,
            AudioLDMPipeline,
            CLIPImageProjection,
            CycleDiffusionPipeline,
            IFImg2ImgPipeline,
            IFImg2ImgSuperResolutionPipeline,
            IFInpaintingPipeline,
            IFInpaintingSuperResolutionPipeline,
            IFPipeline,
            IFSuperResolutionPipeline,
            ImageTextPipelineOutput,
            Kandinsky3Img2ImgPipeline,
            Kandinsky3Pipeline,
            KandinskyCombinedPipeline,
            KandinskyImg2ImgCombinedPipeline,
            KandinskyImg2ImgPipeline,
            KandinskyInpaintCombinedPipeline,
            KandinskyInpaintPipeline,
            KandinskyPipeline,
            KandinskyPriorPipeline,
            KandinskyV22CombinedPipeline,
            KandinskyV22ControlnetImg2ImgPipeline,
            KandinskyV22ControlnetPipeline,
            KandinskyV22Img2ImgCombinedPipeline,
            KandinskyV22Img2ImgPipeline,
            KandinskyV22InpaintCombinedPipeline,
            KandinskyV22InpaintPipeline,
            KandinskyV22Pipeline,
            KandinskyV22PriorEmb2EmbPipeline,
            KandinskyV22PriorPipeline,
            LatentConsistencyModelImg2ImgPipeline,
            LatentConsistencyModelPipeline,
            LDMTextToImagePipeline,
            MusicLDMPipeline,
            PaintByExamplePipeline,
            PixArtAlphaPipeline,
            SemanticStableDiffusionPipeline,
            ShapEImg2ImgPipeline,
            ShapEPipeline,
            StableDiffusionAdapterPipeline,
            StableDiffusionAttendAndExcitePipeline,
            StableDiffusionControlNetImg2ImgPipeline,
            StableDiffusionControlNetInpaintPipeline,
            StableDiffusionControlNetPipeline,
            StableDiffusionDepth2ImgPipeline,
            StableDiffusionDiffEditPipeline,
            StableDiffusionGLIGENPipeline,
            StableDiffusionGLIGENTextImagePipeline,
            StableDiffusionImageVariationPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInpaintPipeline,
            StableDiffusionInpaintPipelineLegacy,
            StableDiffusionInstructPix2PixPipeline,
            StableDiffusionLatentUpscalePipeline,
            StableDiffusionLDM3DPipeline,
            StableDiffusionModelEditingPipeline,
            StableDiffusionPanoramaPipeline,
            StableDiffusionParadigmsPipeline,
            StableDiffusionPipeline,
            StableDiffusionPipelineSafe,
            StableDiffusionPix2PixZeroPipeline,
            StableDiffusionSAGPipeline,
            StableDiffusionUpscalePipeline,
            StableDiffusionXLAdapterPipeline,
            StableDiffusionXLControlNetImg2ImgPipeline,
            StableDiffusionXLControlNetInpaintPipeline,
            StableDiffusionXLControlNetPipeline,
            StableDiffusionXLImg2ImgPipeline,
            StableDiffusionXLInpaintPipeline,
            StableDiffusionXLInstructPix2PixPipeline,
            StableDiffusionXLPipeline,
            StableUnCLIPImg2ImgPipeline,
            StableUnCLIPPipeline,
            StableVideoDiffusionPipeline,
            TextToVideoSDPipeline,
            TextToVideoZeroPipeline,
            TextToVideoZeroSDXLPipeline,
            UnCLIPImageVariationPipeline,
            UnCLIPPipeline,
            UniDiffuserModel,
            UniDiffuserPipeline,
            UniDiffuserTextDecoder,
            VersatileDiffusionDualGuidedPipeline,
            VersatileDiffusionImageVariationPipeline,
            VersatileDiffusionPipeline,
            VersatileDiffusionTextToImagePipeline,
            VideoToVideoSDPipeline,
            VQDiffusionPipeline,
            WuerstchenCombinedPipeline,
            WuerstchenDecoderPipeline,
            WuerstchenPriorPipeline,
        )

    try:
        if not (is_paddle_available() and is_paddlenlp_available() and is_k_diffusion_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_paddle_and_paddlenlp_and_k_diffusion_objects import *  # noqa F403
    else:
        from .pipelines import StableDiffusionKDiffusionPipeline

    try:
        if not (is_paddle_available() and is_paddlenlp_available() and is_fastdeploy_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_paddle_and_paddlenlp_and_fastdeploy_objects import *  # noqa F403
    else:
        from .pipelines import (
            FastDeployCycleDiffusionPipeline,
            FastDeployStableDiffusionControlNetPipeline,
            FastDeployStableDiffusionImg2ImgPipeline,
            FastDeployStableDiffusionInpaintPipeline,
            FastDeployStableDiffusionInpaintPipelineLegacy,
            FastDeployStableDiffusionMegaPipeline,
            FastDeployStableDiffusionPipeline,
            FastDeployStableDiffusionUpscalePipeline,
            FastDeployStableDiffusionXLImg2ImgPipeline,
            FastDeployStableDiffusionXLInpaintPipeline,
            FastDeployStableDiffusionXLInstructPix2PixPipeline,
            FastDeployStableDiffusionXLPipeline,
        )

    try:
        if not (is_paddle_available() and is_librosa_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_paddle_and_librosa_objects import *  # noqa F403
    else:
        from .pipelines import AudioDiffusionPipeline, Mel

    try:
        if not (is_paddlenlp_available() and is_paddle_available() and is_note_seq_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_paddle_and_paddlenlp_and_note_seq_objects import *  # noqa F403
    else:
        from .pipelines import SpectrogramDiffusionPipeline

    try:
        if not (is_note_seq_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_note_seq_objects import *  # noqa F403
    else:
        from .pipelines import MidiProcessor

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
