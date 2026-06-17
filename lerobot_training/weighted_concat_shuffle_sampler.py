from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class WeightedConcatShuffleSampler(Sampler[int]):
    """
    Sample ConcatDataset child datasets by ratio using per-dataset shuffles.

    Dataset ids are sampled with replacement according to ``sample_ratios``. Once a dataset
    id is chosen, the next local index is taken from that dataset's current shuffle order.
    When a dataset's shuffle is exhausted, only that dataset is reshuffled; other datasets
    continue from their current shuffle position.
    """

    _PERMUTATION_SEED_STRIDE = 1_000_003

    def __init__(
        self,
        datasets: Sequence[torch.utils.data.Dataset],
        sample_ratios: Sequence[float],
        num_samples: int | None = None,
        seed: int = 42,
    ):
        if len(datasets) != len(sample_ratios):
            raise ValueError(
                f"Expected one sample ratio per dataset, got {len(sample_ratios)} ratios "
                f"for {len(datasets)} datasets."
            )

        self.lengths = [len(dataset) for dataset in datasets]
        if any(length <= 0 for length in self.lengths):
            raise ValueError(f"All datasets must be non-empty, got lengths {self.lengths}.")

        ratio_tensor = torch.as_tensor(sample_ratios, dtype=torch.float64)
        if not torch.isfinite(ratio_tensor).all():
            raise ValueError(f"Dataset sample ratios must be finite, got {sample_ratios}.")
        if (ratio_tensor < 0).any():
            raise ValueError(f"Dataset sample ratios must be non-negative, got {sample_ratios}.")
        if ratio_tensor.sum().item() <= 0:
            raise ValueError(f"At least one dataset sample ratio must be positive, got {sample_ratios}.")

        self.probabilities = ratio_tensor / ratio_tensor.sum()
        self.cumulative_offsets = [0]
        for length in self.lengths[:-1]:
            self.cumulative_offsets.append(self.cumulative_offsets[-1] + length)

        self.num_samples = sum(self.lengths) if num_samples is None else num_samples
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}.")

        self.seed = seed

        self.dataset_positions = [0] * len(self.lengths)
        self.dataset_shuffle_epochs = [0] * len(self.lengths)
        self._permutations = [self._make_permutation(dataset_idx) for dataset_idx in range(len(self.lengths))]
        self._selection_generator = torch.Generator()
        self._selection_generator.manual_seed(self.seed)
        self.total_samples_yielded = 0

    def __len__(self) -> int:
        return self.num_samples

    def _make_permutation(self, dataset_idx: int) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(
            self.seed
            + dataset_idx * self._PERMUTATION_SEED_STRIDE
            + self.dataset_shuffle_epochs[dataset_idx]
        )
        return torch.randperm(self.lengths[dataset_idx], generator=generator)

    def _reshuffle_dataset(self, dataset_idx: int) -> None:
        self.dataset_shuffle_epochs[dataset_idx] += 1
        self._permutations[dataset_idx] = self._make_permutation(dataset_idx)
        self.dataset_positions[dataset_idx] = 0

    def _rebuild_permutations(self) -> None:
        self._permutations = [
            self._make_permutation(dataset_idx) for dataset_idx in range(len(self.lengths))
        ]

    def _yield_next_index(self) -> int:
        segment_index = int(
            torch.multinomial(self.probabilities, 1, generator=self._selection_generator).item()
        )
        position = self.dataset_positions[segment_index]
        local_index = int(self._permutations[segment_index][position].item())
        self.dataset_positions[segment_index] = position + 1

        if self.dataset_positions[segment_index] >= self.lengths[segment_index]:
            self._reshuffle_dataset(segment_index)

        self.total_samples_yielded += 1
        return self.cumulative_offsets[segment_index] + local_index

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.num_samples):
            yield self._yield_next_index()

    def state_dict(self) -> dict:
        return {
            "dataset_positions": list(self.dataset_positions),
            "dataset_shuffle_epochs": list(self.dataset_shuffle_epochs),
            "selection_generator_state": self._selection_generator.get_state(),
            "total_samples_yielded": self.total_samples_yielded,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.dataset_positions = [int(position) for position in state_dict["dataset_positions"]]
        self.dataset_shuffle_epochs = [
            int(shuffle_epoch) for shuffle_epoch in state_dict["dataset_shuffle_epochs"]
        ]
        self._selection_generator.set_state(state_dict["selection_generator_state"])
        self.total_samples_yielded = int(state_dict["total_samples_yielded"])
        self._rebuild_permutations()
