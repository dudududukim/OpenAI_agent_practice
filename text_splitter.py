import re
from typing import Callable

def cumpa_splitter(
    min_sentence_length: int = 10,      # og : 20
) -> Callable[[str], tuple[str, str]]:
    # TBC
    return