# Project guidance for coding agents

## Hard constraints (non-negotiable)
- This PR must be a strict extension of v1.0_baseline: do not change existing behavior.
- Do not introduce unrelated refactors.
- Keep diffs minimal: change only what the PR request requires, preserve file structure..
- Do not remove or reduce logging.
- If a section is marked LOCKED, do not modify it.
- No refactors unless explicitly requested in the PR description.
- Output contract is immutable:
  - Do not change any existing output filenames.
  - Do not change any existing output locations (S3 keys/prefixes).
  - New functionality may only add *additional* outputs, not rename/replace existing ones.


## Review focus
- Look for behavior regressions (output paths, record counts, filtering conditions).
- Flag any change that could cause empty outputs, missing records, or overwritten datasets.
- Call out any risky assumptions about input JSON structure.
