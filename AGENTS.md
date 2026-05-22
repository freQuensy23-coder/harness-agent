# Repository Agent Rules

- Do not skip the pre-commit hook.
- The pre-commit hook is an intentional verification gate for this repository.
- Do not set skip environment variables or use `--no-verify` to bypass it.
- If the hook fails or appears stuck, investigate and fix the underlying issue instead of bypassing the check.
