---
description: Enforce atomic code modification reviews
globs: *
---

# Code Modification Protocol
You are strictly forbidden from using direct file-writing tools (like `write_file`, `edit_file`, or terminal overrides) to modify code files silently. 

When a file edit is needed, you must follow this two-stage admission process:

1. **Propose the Edit First:** Output a clear markdown diff or code block in the chat showing exactly what you intend to change, add, or delete.
2. **Wait for Admission:** Explicitly ask the user: "Do you approve these specific edits?"
3. **Execution:** Only execute the file modification tool *after* the user gives explicit confirmation in the chat. Never bundle the proposal and the execution into a single step where you output "edited file" after the fact.
