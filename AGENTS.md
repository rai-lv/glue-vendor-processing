# Project guidance for coding agents

## Non-negotiables
- Do not introduce unrelated refactors.
- Keep diffs minimal: change only what the PR request requires, preserve file structure..
- Do not remove or reduce logging.
- If a section is marked LOCKED, do not modify it.
- No refactors unless explicitly requested in the PR description.


## Review focus
- Look for behavior regressions (output paths, record counts, filtering conditions).
- Flag any change that could cause empty outputs, missing records, or overwritten datasets.
- Call out any risky assumptions about input JSON structure.
