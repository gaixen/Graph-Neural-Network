````markdown
```
experiments/
├── wl_failures/
├── depth_collapse/
├── long_range_tasks/
└── rewiring_effects/
```

Purpose: experiments designed to falsify hypotheses — show where architectures fail, not just where they improve.

Principles:

- Design experiments that stress specific theoretical failure modes from `expressivity/` and `training_pathologies/`.
- Report negative results and construct minimal baselines that isolate the factor under test.
- Include exact experimental protocols, seeds, and metrics so results are reproducible and falsifiable.

Design principle:

- If an experiment only shows improvement without probing a failure mode, it is not useful for scientific understanding.
````
