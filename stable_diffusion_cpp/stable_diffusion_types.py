from typing import List, Optional, Dict, Union
from typing_extensions import TypedDict, NotRequired, Literal


class StableDiffusionToken(TypedDict):
    id: int
    prob: float
    logprob: float
    pt: float
    pt_sum: float
