# Scoring bridge notes

The active scoring bridge is implemented as a REINVENT4 plugin:

- `../reinvent_plugins/components/comp_mockdock_oracle.py`

It is loaded through `PYTHONPATH` in `run.py` when `reinvent` is launched.
This plugin keeps an in-process `MDOracle` instance per benchmark run so
budget tracking and docking artifacts are preserved across RL steps.

