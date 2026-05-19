# Instructions for AI Agents (Gemini CLI / Assistants)

Welcome! If you are an AI assistant reading this repository on behalf of the user, you must follow these rules to maintain the integrity of our cross-machine persistent memory.

## 1. Context First
**Before writing any code or proposing architectures**, you MUST read `.ai_context/project_context.md`. This file explains the specific "Iterative Refinement" MDP architecture we are using (where actions are Deltas, not absolute values) and outlines the scripts we have already built.

## 2. Maintain the History Log
Because the user switches between a personal PC and a lab PC, it is critical that you record your work.

**Every time you complete a significant task or end a work session, you MUST:**
1. Open `.ai_context/project_context.md`.
2. Scroll to the `## 5. History Log` section.
3. Append a new bullet point with today's date in `[YYYY-MM-DD]` format.
4. Write a brief, 1-2 sentence summary of exactly what files you modified or what features you implemented.

**How to update the file:**
- If you have file editing tools (like `multi_replace_file_content` or standard python file operations), open the file, locate the end of the History Log, and append your text.
- Example entry: `- **[2026-05-20]**: Added a new plotting script to visualize the loss curves. Modified train_custom_pytorch.py to export training metrics to a CSV.`

## 3. Keep it Clean
Do not track large dataset files (`*.npz`, `*.h5`) or model weights (`*.pth`, `*.pt`) in the git history. We already have a `.gitignore` set up for this. If you create new outputs or datasets, ensure they match the ignored patterns.
