git push# Org Threat Surface Toy Model

Toy simulation of an organizational social graph to explore misaligned AI behavior via token-vs-topology trade-offs.

## Installing

In order of basedness:

### Use direnv

Install direnv.

Run `direnv allow`.

Use [nix-direnv](https://github.com/nix-community/nix-direnv) for faster shells.

### Use nix develop

Install nix.

Run `nix develop`.

### Use UV directly

Install UV.

Run `uv sync`.

## Running

Prefix commands with `uv run`, like:

```
uv run simulation.py --mode sweep --n-orgs 100 --repeats 10 --steps 100
```
