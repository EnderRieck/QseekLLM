# Local Verl Patches

`src/post_train/pyproject.toml` expects a local `verl` checkout. The working
tree used for these experiments was based on:

- repository: `https://github.com/verl-project/verl.git`
- commit: `9f73954a` (`[doc] chore: fix verl ascend readme (#6534)`)

Apply `verl-qseek-local-20260616.patch` from the `src/post_train/verl` checkout
to restore the QseekLLM local changes used by the RL scripts:

```bash
cd src/post_train/verl
git apply ../patches/verl-qseek-local-20260616.patch
```

The full `verl` checkout is intentionally not vendored into this repository, so
the parent repo does not contain an embedded `.git` directory or third-party
history.
