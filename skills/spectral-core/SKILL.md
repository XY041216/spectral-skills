---
name: spectral-core
description: Shared runtime dependency for the Spectral Skills GitHub skill-installer layout. Install alongside spectral-reader, spectral-check, spectral-splitter, spectral-preprocess, spectral-feature, spectral-modeling, spectral-optimizer, spectral-report, and spectral-workflow when using direct GitHub skill installation. Do not use as a user-facing analysis skill; use the stage skills or spectral-workflow instead.
---

# Spectral Core Runtime

This is not a user-facing analysis skill. It packages the shared `spectral_core`
Python runtime so direct GitHub skill installation can run the other Spectral
Skills outside the full Codex plugin image.

Use `spectral-workflow` for end-to-end spectral analysis, or use the stage
skills directly:

- `spectral-reader`
- `spectral-check`
- `spectral-splitter`
- `spectral-preprocess`
- `spectral-feature`
- `spectral-modeling`
- `spectral-optimizer`
- `spectral-report`

If a script reports `ModuleNotFoundError: No module named 'spectral_core'`, the
runtime was not installed. Install this `spectral-core` skill next to the other
Spectral Skills, or install the full `spectral-skills` plugin marketplace
bundle instead.
