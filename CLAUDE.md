# Face Recognition System - Claude Code Rules

## Core Architecture
- All models implement `RecognitionModel` interface from `recognition/base.py`
- Use `registry.get("model_name")` - never instantiate models directly
- Register models with `@register_model("name", priority=N)` decorator
- All embeddings must be L2-normalized (enforced in `EmbeddingResult`)

## File Structure
- Adapters: `recognition/*_adapter.py`
- Config: `config/recognition.yaml` (no hardcoded params)
- Tests: `tests/test_*.py`
- Models: `models/*.onnx` (gitignored)

## Critical Rules
1. Thread-safe Registry: all mutations protected by `_lock`
2. Lazy loading: models load on `registry.get()`, not import
3. Type hints required on all public methods
4. Config-driven: params from `AppConfig.get_model_config()`

## Context7 MCP Usage
- `@docs onnxruntime` - pull ONNX docs when implementing adapters
- `@search "RecognitionModel"` - find existing adapter patterns
- `@search "register_model"` - see registration examples

## Error Handling
- `ModelLoadError` - weights/dependencies failed
- `ModelNotFoundError` - not in registry
- `RuntimeError` - model not loaded
- `ValueError` - invalid input format

## Prohibited
- Global mutable state (except Registry singleton)
- `import *` or bare `except:`
- Direct printing (use `logging`)
- Hardcoded paths (use `Path` + config)

## Ask Before
- Changing `ModelInfo.fingerprint` (breaks compatibility)
- Adding heavy dependencies (>100MB)
- Modifying abstract interface methods
```