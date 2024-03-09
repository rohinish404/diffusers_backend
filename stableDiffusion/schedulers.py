from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from enum import Enum

class SCHEDULER_MAP(Enum):
    DDIMScheduler = DDIMScheduler
    DDPMScheduler = DDPMScheduler
    PNDMScheduler = PNDMScheduler
    LMSDiscreteScheduler = LMSDiscreteScheduler
    EulerDiscreteScheduler = EulerDiscreteScheduler
    EulerAncestralDiscreteScheduler = EulerAncestralDiscreteScheduler
    DPMSolverMultistepScheduler = DPMSolverMultistepScheduler