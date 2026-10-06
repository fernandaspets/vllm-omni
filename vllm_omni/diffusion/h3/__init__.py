"""Runtime extensions for MiniMax-H3: quantised collectives and decoder fixes.

Modules
-------
``a2a_wire``        int8 all-to-all transport for the Ulysses exchange
``ar_wire``         int8 tensor-parallel all-reduce
``a2a_qkv_batch``   batched q/k/v all-to-all in a single collective (opt-in)
``nvfp4``           opt-in W4A4 DiT linear
``quant_policy``    per-role quantisation policy, control-file driven
``vae_sm120``       SM12x video-VAE decoder fixes
``comm``            quantisation primitives and Triton kernels shared by the wire codecs

Everything is opt-in and off by default, so a stock run is unaffected. Nothing is imported here on
purpose: importing this package must not pull in torch, triton or the kernel libraries.
"""
