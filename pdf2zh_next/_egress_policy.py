"""Outbound data policy.

Document content may only leave this machine through the Claude Code CLI.
Downloads (models, fonts, a PDF from a user-given link) are fine, but nothing
else may be uploaded or sent. This module is imported first by the package so
that third-party telemetry is off before those libraries are imported.
"""

import os

# Gradio usage analytics and version-check pings.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
# huggingface_hub (imported by Gradio) usage telemetry.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
