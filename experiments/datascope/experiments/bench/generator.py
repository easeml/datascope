from abc import ABC, abstractmethod
from collections import defaultdict, ChainMap
from itertools import product
from typing import List, Hashable, Iterator, Union, Set, Tuple, Dict
import itertools
import random
import math


MAX_SAMPLING_ATTEMPTS = 10000


class ConfigGenerator(ABC):
    """Abstract base class for generating configurations from a given config space.

    Attributes:
        config_space:
            The configuration space.
    """

    def __init__(self, config_space: Dict[str, List[Hashable]]) -> None:
        self.config_space = config_space
        self.invalid_configs: Set[Tuple[Hashable, ...]] = set()

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Hashable]]:
        """Iterates over the configurations in the config space."""
        pass

    def register_invalid_config(self, config: Dict[str, Hashable]) -> None:
        """Registers a configuration as invalid and prevents it from being generated again. Furthermore,
        it helps some generators with keeping track of the number of valid configurations that were generated.

        Args:
            config:
                The configuration to register as invalid. Keys not present in the config space are ignored.
        """
        invalid_config = tuple(config[k] for k in self.config_space.keys())
        self.invalid_configs.add(invalid_config)


class GridConfigGenerator(ConfigGenerator):
    """Generates configurations by iterating over the product of all variables in the config space."""

    def __iter__(self) -> Iterator[Dict[str, Hashable]]:
        keys, values = zip(*self.config_space.items())
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))


class RandomConfigGenerator(ConfigGenerator):
    """Generates random configurations from the config space.

    Attributes:
        sample_size:
            Number of configurations to sample.
        sampled_configs:
            Set of sampled configurations.
        sampled_values:
            Dictionary tracking sampled values for each variable.
        seed:
            Random seed.
    """

    def __init__(self, config_space: Dict[str, List[Hashable]], sample_size: Union[int, float], seed: int = 0) -> None:
        super().__init__(config_space)
        self.sample_size = sample_size
        self.sampled_configs: Set[Tuple[Hashable, ...]] = set()
        self.sampled_values: Dict[str, Set[Hashable]] = defaultdict(set)
        self.random = random.Random(seed)

    def __iter__(self) -> Iterator[Dict[str, Hashable]]:
        total_combinations = math.prod(len(values) for values in self.config_space.values())
        sample_size = (
            math.ceil(self.sample_size * total_combinations)
            if isinstance(self.sample_size, float)
            else self.sample_size
        )

        while (
            len(self.sampled_configs - self.invalid_configs) < sample_size
            and len(self.sampled_configs | self.invalid_configs) < total_combinations
        ):
            config = self._sample_config()
            if config not in self.sampled_configs:
                self.sampled_configs.add(config)
                result = {k: v for k, v in zip(self.config_space.keys(), config)}
                yield result

    def _sample_config(self) -> Tuple[Hashable, ...]:
        sampled_config: List[Hashable] = []

        for key, values in self.config_space.items():
            remaining_values = set(values) - self.sampled_values[key]
            if remaining_values:
                value = self.random.choice(list(remaining_values))
            else:
                value = self.random.choice(values)
            sampled_config.append(value)

        # If the sampled configuration is invalid, we randomly reset values of variables until a valid configuration is
        # found. This is done to avoid getting stuck in a situation where no valid configurations can be sampled.
        sampling_attempts = 0
        while tuple(sampled_config) in self.invalid_configs:
            sampling_attempts += 1
            if sampling_attempts >= MAX_SAMPLING_ATTEMPTS:
                raise ValueError(
                    f"Could not sample a valid configuration after {MAX_SAMPLING_ATTEMPTS} attempts. "
                    "This may happen if the sample size is too large compared to the number of valid configurations."
                )
            for i, (key, values) in enumerate(self.config_space.items()):
                sampled_config[i] = self.random.choice(values)
                if tuple(sampled_config) not in self.invalid_configs:
                    break

        # Register the values of the sampled configuration.
        for key, value in zip(self.config_space.keys(), sampled_config):
            self.sampled_values[key].add(value)

        return tuple(sampled_config)


class CombinedConfigGenerator(ConfigGenerator):
    """Combines configurations from a list of generators operating on disjoint sets of config variables.

    Attributes:
        generators:
            The list of generators.
    """

    def __init__(self, *generators: ConfigGenerator):
        self.generators = generators

    def __iter__(self) -> Iterator[Dict[str, Hashable]]:
        for configs in product(*self.generators):
            yield dict(ChainMap(*reversed(list(configs))))

    def register_invalid_config(self, config: Dict[str, Hashable]):
        for generator in self.generators:
            generator.register_invalid_config(config)
