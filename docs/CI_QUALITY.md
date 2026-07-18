# Code quality & CI — setup guide

This document explains the six quality checks wired into the repo and how to finish the
setup that **cannot** be done from code (they live in GitHub repository settings).

The CI pipeline itself is `.github/workflows/ci.yml`. It runs on every push to `main` and on
every PR targeting `main`.

| Job | Tool | Config file | Merge status |
| --- | --- | --- | --- |
| `unit-tests` | pytest + pytest-cov | `pytest.ini`, `requirements.txt` | **Required** (blocks merge) |
| `lint` | Ruff | `ruff.toml` | Informational for now |
| `type-check` | mypy | `mypy.ini` | Informational for now |
| `security` | pip-audit | `requirements.txt` | Informational for now |
| `pre-commit` | pre-commit | `.pre-commit-config.yaml` | Informational for now |

The informational jobs use `continue-on-error: true`, so a finding never turns `main` red or
blocks a merge — it only reports. Once the codebase is clean (see "Tightening later"), promote
them to required checks.

---

## 1. Coverage report + visible badge

- `unit-tests` runs pytest with `--cov=app --cov-report=xml --cov-report=html`.
- `coverage.xml` is uploaded to Codecov via `codecov/codecov-action@v5`.
- `htmlcov/` is uploaded as a build artifact on every run (download it from the run summary to
  browse line-by-line coverage).
- `codecov.yml` configures Codecov to post a per-PR comment with the coverage delta.
- The visible badge is in `README.md` (top of the file):
  `https://codecov.io/gh/<owner>/<repo>/branch/main/graph/badge.svg`

**To finish (one time, repo settings):** the badge works automatically for a **public** repo with
no token. For a **private** repo, create a Codecov token
(Codecov → your repo → Settings → General → Upload token) and add it as a
`CODECOV_TOKEN` secret under **Settings → Secrets and variables → Actions**. The workflow already
references it; an empty/unset value is fine for public repos.

---

## 2. Linting — Ruff

- Config: [`ruff.toml`](../ruff.toml). Starting rule set: `E, F, I, W, UP, B`
  (pycodestyle errors/warnings, pyflakes, isort, pyupgrade, flake8-bugbear). `B008` is ignored
  because FastAPI `Depends()` triggers it spuriously.
- CI job: `lint` runs `ruff check .` (pinned `ruff==0.6.9`, matching the pre-commit hook).

Run locally:

```bash
pip install ruff==0.6.9
ruff check .                 # report
ruff check . --fix           # auto-fix safe issues (unused imports, import order, ...)
ruff check . --statistics    # grouped counts per rule
```

---

## 3. Type checking — mypy

- Config: [`mypy.ini`](../mypy.ini). Deliberately lenient (no `disallow_untyped_defs`, third-party
  untyped modules silenced) so it reports real inconsistencies in the hints you already have
  without a wall of noise from LangChain / sentence-transformers stubs.
- CI job: `type-check` runs `mypy app`.

Run locally:

```bash
pip install mypy
mypy app
```

---

## 4. Dependency security scan — pip-audit + Dependabot

Two complementary layers:

- **pip-audit (CI):** the `security` job runs `pip-audit -r requirements.txt`, which checks the
  pinned versions against the PyPI / OSV advisory database and fails (red, non-blocking) if a
  known vulnerability is present.
- **Dependabot (`.github/dependabot.yml`):** opens automated PRs to keep `requirements.txt` and the
  GitHub Actions in the workflow current (weekly).

Run pip-audit locally:

```bash
pip install pip-audit
pip-audit -r requirements.txt
```

**To finish (one time, repo settings):** enable **Dependabot security updates** and **Dependabot
alerts** under **Settings → Code security → Dependabot**. For public repos these are on by default;
for private repos toggle them on. This makes GitHub open fix PRs automatically when a new
CVE is published for a dependency you use.

---

## 5. Pre-commit hooks

- Config: [`.pre-commit-config.yaml`](../.pre-commit-config.yaml). Hooks: `trailing-whitespace`,
  `end-of-file-fixer`, `check-yaml`, `check-added-large-files` (max 1024 KB), `check-merge-conflict`,
  `debug-statements`, and `ruff`.
- The **same hooks run in CI** via the `pre-commit` job (`pre-commit/action@v3.0.1`, which runs
  `pre-commit run --all-files`), so they cannot be skipped by pushing from a machine without the
  hook installed.
- Hooks run only on **changed files**, so installing pre-commit does not block commits to
  untouched files.

Install (one time, on every clone):

```bash
pip install pre-commit
pre-commit install
```

After that, every `git commit` runs the hooks automatically. To check the whole repo once:

```bash
pre-commit run --all-files
```

To upgrade the hook tool versions later: `pre-commit autoupdate`.

---

## 6. Branch protection (manual, repo settings)

This **cannot** be configured from a file — it must be set in GitHub.

1. Go to **Settings → Branches → Branch protection rules → Add rule**.
2. **Branch name pattern:** `main`.
3. Enable **Require a pull request before merging** (optional: require approvals).
4. Enable **Require status checks to pass before merging**.
5. Under "Status checks that are required", search for and select **`unit-tests`** (the job name
   from `ci.yml`). This is the only job that currently blocks merges — exactly as intended.
6. (Recommended) Enable **Require branches to be up to date before merging** so stale PRs can't
   sneak in.
7. Save.

Now no one — including you, by accident — can merge a PR into `main` while `unit-tests` is red.

> When you later clean up the lint/type-check/security findings, return here and also add those
> job names (`lint`, `type-check`, `security`, `pre-commit`) as required checks to enforce them.

---

## Tightening later (suggested order)

1. Address Ruff findings: `ruff check . --fix` clears the safe ones; fix the rest by hand.
   Then promote `lint` to a required check.
2. Promote `security` to required once `pip-audit` is clean.
3. Promote `type-check` after tightening `mypy.ini` (e.g. `disallow_untyped_defs = True` on leaf
   modules).
4. Consider adding more Ruff rule families (`C4`, `SIM`, `RUF`, `PTH`, `N`, ...) in `ruff.toml`.
