# Data Pipeline Scope Decisions

## Ground-Truth Database Sampling

The pinned configuration supplies a database-sampler mapping whose `enable` field is `False`.
However, the pinned preprocessing constructor checks only whether that mapping is `None`, and its
sampler builder does not consume `enable`. Literal execution can therefore sample database objects
despite the configuration's disabled flag.

The frozen local baseline follows the declared `enable=False` intent and does not implement or run
ground-truth database sampling. Adding it would require a new reviewed baseline decision and parity
fixtures rather than an implicit change to training data.

## Randomness And Mutation

Augmentation and sweep loading accept an injectable NumPy-compatible RNG while defaulting to
`numpy.random`. The default draw order and float32 assignment behavior match the pinned helpers;
injection exists so tests can prove that order without changing global process state.

Local augmentation returns transformed copies instead of mutating caller-owned arrays. This does
not change produced points or boxes, but prevents sample-cache aliasing from becoming part of the
public data interface.
