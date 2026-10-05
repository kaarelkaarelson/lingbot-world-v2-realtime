"""How one run is laid out across GPUs. One GPU is just degree 1."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelConfig:
    sp: int = 1  # Ulysses sequence parallel: DiT tokens split across `sp` GPUs, all-to-all around attention

    @property
    def world_size(self):
        return self.sp

    def validate(self, num_heads, world_size):
        if self.world_size != world_size:
            raise ValueError(f"{self} needs {self.world_size} processes, got {world_size}")
        if num_heads % self.sp:
            raise ValueError(f"sequence parallel degree {self.sp} must divide the {num_heads} attention heads")
