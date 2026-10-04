"""MiniMax-H3 lane runtime extensions.

These modules previously shipped as loose files on ``PYTHONPATH`` and were imported by bare name. They
live in the tree so a checkout of this branch is self-contained: no out-of-tree shim directory is
required at runtime.

Contents
--------
``a2a_wire``        int8 all-to-all transport (the 0.5625x byte lever)
``ar_wire``         int8 all-reduce transport
``a2a_qkv_batch``   batched/stacked qkv all-to-all (opt-in arm; measured slower, default off)
``nvfp4``           opt-in W4A4 DiT linear
``quant_policy``    per-role quantisation policy, control-file driven
``vae_sm120``       SM12x VAE decoder fixes
``comm``            quantisation primitives and kernels shared by the wire codecs

Nothing is imported here on purpose: importing this package must not pull in torch, triton or the
kernel libraries.
"""
