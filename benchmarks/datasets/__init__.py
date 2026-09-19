"""One module per benchmark: how to read it, and how it is scored.

A benchmark differs from another in two places only -- how a conversation is
laid out, and which metric answers its question. Everything else (the arms,
the shared model, the fairness assertions, the resource accounting) is common,
so adding a benchmark is adding a module here.
"""
