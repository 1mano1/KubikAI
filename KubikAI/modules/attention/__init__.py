from typing import *

# Hardcode the backend to 'sdpa' to use the built-in PyTorch implementation
# and avoid dependencies on xformers or flash_attn.
BACKEND = 'sdpa'
DEBUG = False

print(f"[ATTENTION] Using backend: {BACKEND}")

def set_backend(backend: Literal['xformers', 'flash_attn', 'sdpa', 'naive']):
    global BACKEND
    BACKEND = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

from .full_attn import *
from .modules import *
