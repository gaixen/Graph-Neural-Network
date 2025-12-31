````markdown
```
utils/
├── __init__.py
```

Purpose: boring but necessary utilities: logging, config, reproducibility helpers, and small tools used across the repo.

Contents and expectations:

- Seeding and reproducibility utilities with documented limitations on platform-dependent randomness.
- Logging helpers and experiment config loaders (clear, minimal, documented defaults).
- No conceptual content: this folder should not contain model code or theory.

Design principle:

- Keep utilities minimal and explicit. Prefer small, well-documented helpers over large, opinionated toolboxes.
````
