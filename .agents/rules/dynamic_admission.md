# Rule: Granular Dynamic Admission (Per-Row) for Python Code Edits

## Context
The user wants granular, line-by-line (per-row) control over all proposed changes to Python (`.py`) files. They want to be able to accept or reject specific lines of code individually.

## Instruction
Before modifying or creating any Python (`.py`) file:
1. **Stop execution immediately.**
2. Format the proposed edits as a **numbered list of individual line changes (rows)**, showing exactly what is being added, removed, or modified.
3. Explicitly ask the user to specify which rows (by their number) they approve or reject.
4. **Do not run the tool** until the user replies.
5. Only execute the edits for the specific rows/lines that the user explicitly approved.
