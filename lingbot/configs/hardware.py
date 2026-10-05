"""Default parallel layout per (GPU model, GPU count). New hardware is a new row, not new code."""
from lingbot.parallel.config import ParallelConfig

HARDWARE = {
    ("rtx5090", 1): ParallelConfig(sp=1),
    ("rtx5090", 2): ParallelConfig(sp=2),  # PCIe 5.0 x16, no NVLink
}


def default_parallel(gpu, count):
    try:
        return HARDWARE[(gpu, count)]
    except KeyError:
        known = ", ".join(f"{g} x{n}" for g, n in HARDWARE)
        raise KeyError(f"no parallel layout for {gpu} x{count}; known: {known}") from None
