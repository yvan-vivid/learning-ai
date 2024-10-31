# Karpathy Series Exercises

This is my effort to work through the Karpathy *zero to hero* series, attempting to formulate components in a library with decent python software engineering rather than just compiling everything *ad-hoc* in notebooks. The notebooks import the libraries and implement their components.

## Implemented 

### Micrograd

I did a very different implementation of this, since it was foundational and I had some better ideas about how I wanted to do it differently.

### Makemore

I have done the first two parts of this so far.

## Development

To do development on this, I use **Poetry**, **Poe the Poet**, and **Python 3.12**. These are defined in the parent directory in the `shell.nix`. I use **NixOS** and thus can build a local environment using **Lori** and **direnv**. If you don't have **Nix**, you can, of course, simply have these three requirements installed in whatever environment you have. Once that is set
```bash
poetry install
```
creates the venv and installs everything else, including all the other build and test tools.

```bash
poe check
```
will run the type checkers (**mypy** and **basedpyright**), the tests, and lint with **ruff**.

There are a number of other `poe` targets. I use `poe edit`, which just starts **neovim** in the virtual environment so my LSP setup will use the servers therein.

There are several notebooks in the project for the actual exercises. Running
```bash
poe notebooks
```
will start `jupyterlab` in the virtual environment, where they should have access to the libraries.


