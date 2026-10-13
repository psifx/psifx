"""Shared runtime constants."""

import os


# Allow overriding the default SAM3 model path without editing source files.
SAM3_PATH = os.environ.get("SAM3_PATH", "facebook/sam3")
