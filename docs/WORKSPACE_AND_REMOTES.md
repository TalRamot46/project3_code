# project3 workspace and Git remotes

## Default workspace (code + LaTeX)

To get **both** the code and the LaTeX docs in one place (for context and physical background):

1. Open the **workspace file** (don’t open only the project_3 folder):
   - **File → Open Workspace from File…**
   - Choose: `project_3\project3.code-workspace`
   - Or double‑click `project3.code-workspace` in Explorer.

2. The workspace **name** is **project3**. It contains two roots:
   - **project3_code** — this repo (simulation code: hydro_sim, rad_hydro_sim, shussman_solvers, etc.).
   - **project3_docs** — the LaTeX repo (Numerical_Problems clone). Project‑3 LaTeX is in `project3_docs/project_3/`.

Your LaTeX shortcuts (e.g. Ctrl+S to build/refresh PDF) work in the docs folder as before.

---

## Git remotes (project3_code and project3_docs)

So you can tell them apart, use these remote **names**:

| Repo        | Remote name     | Where |
|------------|------------------|--------|
| Code       | **project3_code** | In `project_3` (this repo). |
| LaTeX/docs | **project3_docs** | In the `project3_docs` clone. Already set. |

### Docs repo (project3_docs)

In the clone at `C:\Users\TLP-001\Documents\GitHub\project3_docs`, the remote is already named **project3_docs** (Overleaf). Nothing else to do there.

### Code repo (project3_code)

This repo (project_3) currently has **no** remote. When you add your code host (e.g. GitHub):

```bash
cd C:\Users\TLP-001\Documents\GitHub\project_3
git remote add project3_code <YOUR_CODE_REPO_URL>
```

Use `project3_code` as the remote name (not `origin`) so it matches the workspace folder name. To push/pull:

```bash
git push project3_code main
git pull project3_code main
```

If you prefer to keep using `origin` for the code repo, you can add it as `origin` and add a second remote `project3_code` pointing to the same URL; the workspace folder name is only for display.

---

## Summary

| What | Where / command |
|------|-------------------|
| Open default workspace (code + docs) | **File → Open Workspace from File…** → `project_3\project3.code-workspace` |
| Workspace name | **project3** (two roots: project3_code, project3_docs) |
| Code remote | Add when you have a URL: `git remote add project3_code <url>` |
| Docs remote | Already **project3_docs** in `project3_docs` clone |
| Project‑3 LaTeX | `project3_docs/project_3/` (rad_hydro_sim, Shussman, etc.) |
