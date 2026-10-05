# WEIGHTS — what to download, from where, and what is optional

The image contains **no model weights**. They are a ~270 GB download and belong in a cache, not an
image layer. Everything below is a public Hugging Face repository except the RefMods, which are
explicitly optional.

## Required

| what | Hugging Face repo | notes |
|---|---|---|
| **base model** | [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) | public, ungated. Diffusers-format MiniMax-H3. ~269 GB. |
| **4-step turbo LoRA** | [`lightx2v/Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | public. Use `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` — this is the standing 4-step recipe. |

### The `ref2va` config

Serving uses `--task-type ref2va`, which points at a directory that is a small `model_index.json`
plus a `Ref2VA` component. The lane's copy is:

```
/mnt/2king/models/MiniMaxAI/MiniMax-H3-ref2va/
├── model_index.json        # the ref2va composition
└── Ref2VA -> ../MiniMax-H3/Ref2VA
```

`Ref2VA` already ships inside `MiniMaxAI/MiniMax-H3`, so this is a 4 KB wrapper plus a symlink, not a
separate download. The image's `scripts/ref2va/` holds the wrapper; point it at your
`MiniMax-H3/Ref2VA` and the lane serves.

## Optional — identity RefMods

| what | where |
|---|---|
| `minimaxh3_rocco_v2_refmod.safetensors`, `minimaxh3_roxy_v2_refmod.safetensors` | **not published** — produced locally from the pack's reference photos |

They are identity adapters for the two dogs. **The lane boots and renders without them**; what is
lost is character consistency between shots — Rocco and Roxy come from the scene description and the
reference images instead of a fixed identity.

To publish them, upload the two `*_v2_refmod.safetensors` files to a Hugging Face repo and set
`H3_REFMOD_PATHS` to that repo's snapshot path. Until then, document them as required-for-consistency,
not required-to-run.

## Pointing the lane at the weights

```bash
# default: local paths, as the lane has always run
H3_WEIGHTS_SOURCE=local

# or: resolve from Hugging Face into the cache
H3_WEIGHTS_SOURCE=hf
```

`scripts/h3_fetch_weights.py` handles the `hf` path (repo ids and revisions from the table above).
The acceptance test for the `hf` path is that it renders a clip **byte-identical** to the `local` path
on the same seed; that has not yet been demonstrated, so treat `hf` as convenience, not as verified
equivalence.

## Sizing

| | |
|---|---|
| weights on disk | ~272 GB (model + LoRA) |
| cache volume | mount a large volume at `/root/.cache/huggingface` |
| host RAM | the lane parks in host RAM when asleep (`enable_sleep_mode`), so leave headroom |
| GPUs | 4 × SM120 (RTX PRO 6000 class), TP2 × USP2. The lane does not fit on 32 GB cards. |
