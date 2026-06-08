"""Streamlit demo package for the Multi-Agent MCP System.

Split into:
  - app.py        — entrypoint / orchestration
  - ui.py         — Streamlit frontend (page chrome, welcome, trace view, switch)
  - backend.py    — pipeline, inventory, scope resolution, trace building (no UI)
  - log_stream.py — log handlers bridging backend logs to UI placeholders
  - static/       — styles.css
"""