# Agent Instructions

Before starting or resuming any training, evaluation, data split, preprocessing,
or benchmark task in this repository, every model or agent MUST:

1. Read `protocol.md` in full.
2. Check the requested experiment against every applicable protocol rule.
3. Run the required pre-training data integrity checks.
4. Stop and report any conflict instead of silently changing the protocol,
   dataset split, task definition, or fixed hyperparameters.

These requirements apply on every training run, including resumed and delegated
runs. User instructions take precedence only when the user explicitly approves
a protocol deviation; that deviation must be recorded in the experiment output.

