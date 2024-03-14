from typing import List, Optional, Dict, Union
from typing_extensions import TypedDict, NotRequired, Literal


class StableDiffusionToken(TypedDict):
    id: int
    prob: float
    logprob: float
    pt: float
    pt_sum: float


class StableDiffusionSegment(TypedDict):
    start: int
    end: int
    text: str
    tokens: List[StableDiffusionToken]


class StableDiffusionResult(TypedDict):
    task: Literal["transcribe", "translate"]
    language: str
    duration: float
    segments: List[StableDiffusionSegment]
    text: str
