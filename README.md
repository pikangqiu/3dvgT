# VggT

Satellite-guided and BEV-guided 3D reconstruction research scaffold.

This project uses G3T as the main gravity-aligned reconstruction reference and uses the Look from Above paper as the main nuScenes satellite/BEV alignment paradigm reference.

See `agent.md` for the research纲领 and working rules.

## Layout

```text
.
├── agent.md
├── configs/
├── docs/
├── refs/
│   ├── g3t/
│   └── look-from-above/
├── scripts/
└── src/
```

`refs/g3t` is a cloned reference repository. `refs/look-from-above` currently stores notes because no official public code repository has been identified for the anonymous Look from Above submission. `refs/look-from-above-components` contains public component references such as PseudoMapTrainer and MapTR.
