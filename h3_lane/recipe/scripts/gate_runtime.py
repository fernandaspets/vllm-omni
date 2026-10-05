#!/usr/bin/env python3
"""Fail the build if the install moved a package the base runtime shipped.

The dependency lock installs ONLY packages the runtime does not have, with --no-deps, so the set of
distribution versions the base image started with must survive the build unchanged. This gate is the
check `pip check` cannot make: `pip check` reports the runtime's own pre-existing conflicts (the bare
base fails it with the wandb/opentelemetry-api line) and the --no-deps install of vllm-omni adds
"requires X, not installed" lines by construction, so it fails for reasons that are not defects.
What must actually hold is that nothing in runtime-dist.json changed.

Intentional exceptions live in ALLOW (package -> why it is expected to differ).
"""
import importlib.metadata as md
import json
import sys

# Packages this recipe replaces on purpose. Everything else must be untouched.
ALLOW = {
    "b12x": "installed from the pinned PR branch (the runtime ships an older 1.3.0)",
}


def main(path: str) -> int:
    manifest = json.load(open(path))
    changed, removed, allowed = [], [], []
    for name, want in sorted(manifest.items()):
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            removed.append(f"{name}=={want}")
            continue
        if got != want:
            (allowed if name in ALLOW else changed).append(f"{name} {want} -> {got}")
    print(f"runtime gate: {len(manifest)} base packages checked, {len(changed)} changed, "
          f"{len(removed)} removed, {len(allowed)} allowed")
    for line in allowed:
        print(f"  allowed  {line}   [{ALLOW[line.split()[0]]}]")
    for line in changed:
        print(f"  CHANGED  {line}")
    for line in removed:
        print(f"  REMOVED  {line}")
    if changed or removed:
        print(f"FAIL: the install perturbed the runtime "
              f"({len(changed)} changed, {len(removed)} removed)", file=sys.stderr)
        return 1
    print("runtime gate: ok - every package the base shipped is at its original version")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
