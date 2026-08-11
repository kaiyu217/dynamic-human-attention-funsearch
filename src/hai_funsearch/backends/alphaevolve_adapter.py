"""AlphaEvolve-on-Google-Cloud integration notes.

The public client library exposes candidate programs, per-candidate metrics, and insights through
its experiment/controller API. A production adapter should intercept candidate handling around the
local evaluator/controller loop, compute `(promise, uncertainty)`, query the attention policy, and
submit either normal evaluator feedback or human-targeted insights.

This file avoids importing Google Cloud dependencies so the core package stays runnable offline.
"""
