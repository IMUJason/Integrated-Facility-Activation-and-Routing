# Repository Scope

This repository was curated from the local project workspace to avoid version mismatch and accidental release of restricted inputs.

Included:

- aggregated result tables and JSON summaries corresponding to the current revision
- solver code that runs from the released benchmark instances
- anonymized benchmark instances derived from the retained Quzhou case

Excluded:

- current manuscript source files, supplementary files, highlights, and compiled submission PDFs
- raw administrative filings and fleet records
- geocode override files and cleaned project/site records with original identifiers
- old TRSC submission assets and proof files
- obsolete `v1`, `v2`, and backup experiment packages
- local build artifacts, cached bytecode, and machine-specific paths

Practical consequence:

- the public release supports manuscript-level reproducibility from the released benchmark instances onward
- it does not provide a redistributable path back to the restricted administrative source records
