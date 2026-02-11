# Documentation and LaTeX (Numerical problems)

The **LaTeX write-up** for this project lives in a separate Git repository called **"Numerical problems"**, which is shared across several projects. It is integrated here as a **Git submodule** so that:

- The LaTeX repo keeps its **full commit history** and stays a separate repo (you can still clone/push it on its own).
- This project only stores a **reference** (path + commit) to the submodule; no copy of the .tex files is kept in project_3 history.
- Opening this workspace (project_3) together with the submodule lets you and tooling (e.g. AI) see both code and write-up for context and physical background.

You already have editor shortcuts (e.g. Ctrl+S to save and refresh the LaTeX PDF preview); no change needed for that.

---

## One-time setup (already done)

The **Numerical problems** repo (Overleaf) is already added as a submodule at **docs/Numerical problems**. A separate full clone lives at **project3_docs** (sibling of project_3) for the default workspace. See **docs/WORKSPACE_AND_REMOTES.md** for opening the **project3** workspace (code + docs) and for Git remote names (**project3_code** / **project3_docs**).

If you need to re-add the submodule elsewhere (e.g. after a fresh clone without submodules), use the same remote URL:

```bash
git submodule add "https://git.overleaf.com/692577a6bc1fa232094e71a3" "docs/Numerical problems"
```

---

## Cloning this repo with the LaTeX docs

After the submodule is added, anyone (or you on another machine) should clone with submodules so the "Numerical problems" folder is populated:

```bash
git clone --recurse-submodules <project_3_repo_url>
```

If the repo was already cloned without submodules:

```bash
git submodule update --init --recursive
```

Or from project_3: `make submodule-init` (see Makefile).

---

## Working with the LaTeX (submodule) repo

- **Edit and build LaTeX** inside `docs/Numerical problems/` as usual (your Ctrl+S / preview setup is unchanged).
- **Commit LaTeX changes in the submodule only:**
  ```bash
  cd "docs/Numerical problems"
  git add .
  git commit -m "Update write-up ..."
  git push   # if the Numerical problems repo has a remote
  ```
- **Record the new doc version in project_3** (optional): from project_3 root, run `git add "docs/Numerical problems"` and commit. That updates the “pointer” to the LaTeX repo so this project references the latest doc commit.

You do **not** need a separate branch in project_3 for the .tex files. The usual practice is to keep the submodule on your main branch; all history stays in the "Numerical problems" repo.

---

## Summary

| What | Where |
|------|--------|
| LaTeX source | `docs/Numerical problems/` (submodule; own Git history) |
| Add submodule (one-time) | `git submodule add <path-or-url> "docs/Numerical problems"` from project_3 |
| Clone with docs | `git clone --recurse-submodules ...` or `git submodule update --init --recursive` |
| Commit .tex changes | Inside `docs/Numerical problems` (submodule repo) |
| Optional: pin new doc version in project_3 | `git add "docs/Numerical problems"` and commit in project_3 |
