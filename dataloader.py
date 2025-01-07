from pathlib import Path # 문자열이 아니라, 경로로서 다룬다.
from collections import Counter
import random
from copy import deepcopy
import numpy as np
import torch
from datasets import IterableDataset
from datasets.iterable_dataset import RandomlyCyclingMultiSourcesExamplesIterable
from datasets.utils.logging import get_logger
from datasets import load_from_disk

logger = get_logger(__name__)


RANDOM_BATCH_SIZE = 8192
DEFAULT_SEED=111

# XXX - 얘는 뭐???
class UpdatableRandomlyCyclingMultiSourcesExamplesIterable(
        RandomlyCyclingMultiSourcesExamplesIterable):

    def __init__(self, ex_iterables, generator, probabilities=None, probabilities_handle=None, stopping_strategy="all_exhausted"):
        '''
        probabilities: vector of static probabilities over training
        probabilities_handle: handle to domain weights buffer in model params
        '''
        super().__init__(ex_iterables, generator, stopping_strategy=stopping_strategy)
        self.probabilities_handle = probabilities_handle
        self.probabilities = probabilities

    @staticmethod
    def _iter_random_indices(rng, num_sources, probabilities_handle=None, probabilities=None, random_batch_size=RANDOM_BATCH_SIZE):
        while True:
            # read domain weights
            if probabilities_handle is not None:
                probabilities = probabilities_handle.detach().cpu().numpy()

            yield from (int(i) for i in rng.choice(num_sources, size=random_batch_size, p=probabilities))

    def _give_indice_iterator(self):
        rng = deepcopy(self.generator)
        return self._iter_random_indices(rng, len(self.ex_iterables), probabilities_handle=self.probabilities_handle, probabilities=self.probabilities)

    def shard_data_sources(self, shard_indices):
        return self

    @property
    def n_shards(self):
        return 1

    def shuffle_data_sources(self, seed):
        self.ex_iterables = [ex_iterable.shuffle_data_sources(seed) for ex_iterable in self.ex_iterables]
        return self


def interleave_datasets(datasets, probabilities=None, probabilities_handle=None, seed=None, stopping_strategy='all_exhausted'):
    iterable_datasets = []
    for dataset in datasets:
        if not isinstance(dataset, IterableDataset):
            iterable_datasets.append(dataset.to_iterable_dataset())
        else:
            iterable_datasets.append(dataset)

    ex_iterables = [d._ex_iterable for d in iterable_datasets]

    generator = np.random.default_rng(seed)
    ex_iterable = UpdatableRandomlyCyclingMultiSourcesExamplesIterable(
            ex_iterables, generator=generator,
            probabilities=probabilities, probabilities_handle=probabilities_handle,
            stopping_strategy=stopping_strategy)

    return IterableDataset(ex_iterable=ex_iterable)


def simulate_data_skip_per_domain(num_skip_examples, probabilities, rng, random_batch_size=RANDOM_BATCH_SIZE):
    num_sources = len(probabilities)
    sampled_domain_idxs = [
        rng.choice(num_sources, size=random_batch_size, p=probabilities)
        for _ in range(num_skip_examples // random_batch_size + 1)]
    sampled_domain_idxs = np.concatenate(sampled_domain_idxs)[:num_skip_examples]
    counts = Counter(sampled_domain_idxs)
    return [counts.get(i, 0) for i in range(num_sources)]


def determine_skip_per_domain(num_skip_examples, seed, domain_weights, domain_names):
    if num_skip_examples == 0:
        return {name: 0 for name in domain_names}

    if domain_weights is None or seed is None or domain_names is None:
        raise ValueError("If num_skip_examples > 0 then domain_weights, domain_names, and seed must not be None")

    rng = np.random.default_rng(seed) # random generator
    skip_per_domain = simulate_data_skip_per_domain(num_skip_examples, domain_weights, rng)
    domain_name_to_skip_num = {name: num for name, num in zip(domain_names, skip_per_domain)}
    return domain_name_to_skip_num

# XXX loop
# XXX yield의 장점
def skippable_data_gen(shards, num_skip_examples=0, loop=True, seed=111, shuffle=False, keep_in_memory=False):

    def get_shard_ds(shard_dir, num_skipped, seed, shuffle):
        # TODO hack:
        if keep_in_memory:
            curr_keep_in_memory = (hash(str(shard_dir)) % 2 == 0)
        else:
            curr_keep_in_memory = False

        shard = load_from_disk(dataset_path=str(shard_dir), keep_in_memory=curr_keep_in_memory)
        if shuffle:
            shard = shard.shuffle(seed=seed)
        if num_skipped < num_skip_examples:
            # try to skip examples
            if len(shard) < (num_skip_examples - num_skipped):
                num_skipped += len(shard)
            else:
                shard = shard.select(range(num_skip_examples - num_skipped, len(shard)))
                logger.info(f"Skipped {num_skip_examples} examples in {shard_dir}")
                num_skipped = num_skip_examples
        return shard, num_skipped

    num_skipped = 0
    if loop:
        while True:
            for shard_dir in shards:
                shard, num_skipped = get_shard_ds(shard_dir, num_skipped, seed, shuffle)
                if num_skipped < num_skip_examples:
                    continue

                for ex in shard:
                    yield ex
                seed += 1
    else:
        for shard_dir in shards:
            shard, num_skipped = get_shard_ds(shard_dir, num_skipped, seed, shuffle)
            if num_skipped < num_skip_examples:
                continue

            for ex in shard:
                yield ex
            seed += 1

def get_perdomain_datasets(
        preprocessed_dir,
        domain_weights_dict,
        cache_dir=None,
        split=None,
        seed=DEFAULT_SEED,
        domain_weights=None,
        domain_names=None,
        num_skip_examples=0,
        shuffle=False,
        shard_reversal=False,
        keep_in_memory=False):
    '''
    Returns a dictionary from domain name to IterableDataset.
    '''
    domain_name_to_skip_num = determine_skip_per_domain(num_skip_examples, seed, domain_weights, domain_names) # domain 당 skip할 것을 계산.

    preprocessed_dir = Path(preprocessed_dir)
    if split is not None and (preprocessed_dir / split).exists():
        preprocessed_dir = preprocessed_dir / split
    else:
        logger.warn("No split used or split directory not found: using same data for all splits.")

    domains = list(sorted(domain_weights_dict.keys())) 

    all_ds = {}
    for domain in domains:
        domain_dir = preprocessed_dir / domain

        if (domain_dir / 'dataset_info.json').exists():
            curr_shards = [domain_dir]
        else:
            curr_shards = list(domain_dir.iterdir())
            # shuffle shard order
            random.Random(seed).shuffle(curr_shards)
            if shard_reversal:
                curr_shards = list(reversed(curr_shards))

        ds = IterableDataset.from_generator(
                skippable_data_gen,
                gen_kwargs={'shards': curr_shards,
                            'num_skip_examples': domain_name_to_skip_num[domain],
                            'loop': (split == 'train'),
                            'seed': seed,
                            'shuffle': shuffle,
                            'keep_in_memory': keep_in_memory}
                )
        all_ds[domain] = ds
        seed += 1
    return all_ds


def get_preprocessed_mixed_dataset(
        preprocessed_dir,
        domain_weights_dict,
        cache_dir=None,
        split='train',
        seed=DEFAULT_SEED,
        max_samples=None,
        add_domain_id=False,
        domain_weight_buffer_handle=None,
        tokenizer=None,
        no_interleave=False,
        shuffle=False,
        num_skip_examples=0,
        shard_reversal=False,
        keep_in_memory=False):
    '''preprocessed_dir: has the following format
               first level: domain directories
               second level: shards for each domain. number of shards per domain should be the same. #XXX - why?

       domain_weights_dict: dict from domain name to weight
       dataset_name: name of dataset. update this function to introduce new datasets
       cache_dir: cache directory for arrow files (if needed)
       split: train or validation
       seed: int (controls ordering of data shards)
       max_samples: int (limit for number of examples)
       add_domain_id: add domain id to the bath on the fly
       domain_weight_buffer_handle: handle to the domain weights in the model params 
       tokenizer: huggingface tokenizer
       no_interleave: don't interleave the domains - just iterate through the data in order
       shuffle: on-the-fly shuffle with a buffer size 100k
       num_skip_examples: skip examples in the iterator
       shard_reversal: reverse the shard ordering to prioritize unseen examples
    '''
    domain_names = list(sorted(domain_weights_dict.keys()))
    domain_to_idx = {domain_names[i]: i for i in range(len(domain_names))}
    domain_weights = np.asarray([domain_weights_dict[domain_name] for domain_name in domain_names])
    domain_weights = domain_weights / domain_weights.sum()

    probabilities = domain_weights

    # TODO: load pile with perdomain_datasets
#     if dataset_name == 'pile':
#         all_ds = get_pile_datasets(
#                 preprocessed_dir,
#                 cache_dir=cache_dir,
#                 split=split,
#                 seed=seed,
#                 domain_weights=domain_weights, # XXX
#                 domain_names=domain_names, # XXX
#                 num_skip_examples=num_skip_examples, # XXX -> train에서 확인할 것.
#                 shuffle=shuffle,
#                 shard_reversal=shard_reversal,
#                 keep_in_memory=keep_in_memory)
#     else:
    try:
        all_ds = get_perdomain_datasets(
            preprocessed_dir, 
            domain_weights_dict,
            cache_dir=cache_dir,
            split=split,
            seed=seed,
            domain_weights=domain_weights,
            domain_names=domain_names,
            num_skip_examples=num_skip_examples,
            shuffle=shuffle,
            shard_reversal=shard_reversal,
            keep_in_memory=keep_in_memory)
    except Exception:
        raise ValueError(f"dataset_name {dataset_name} not implemented.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
#########################################################################################################################################################
    def add_domain_id_generator(ds, domain_idx):
        for ex in ds:
            ex['domain_id'] = domain_idx
            yield ex
    domain_ds_ls = []
    for domain_name in domain_names:
        domain_idx = domain_to_idx[domain_name]
        domain_ds = all_ds[domain_name]
        # add domain_id if necessary
        if add_domain_id:
            domain_ds = IterableDataset.from_generator(add_domain_id_generator, gen_kwargs={'ds': domain_ds, 'domain_idx': domain_idx})
        domain_ds_ls.append(domain_ds)

    if no_interleave:
        # instead of interleaving, run through each dataset
        def data_generator(shards):
            for shard in shards:
                for ex in shard:
                    yield ex
        ds = IterableDataset.from_generator(data_generator, gen_kwargs={'shards': domain_ds_ls})
        logger.info("Not interleaving dataset - will not sample according to domain weights")

    else:
        # interleave - 뒤섞다.
        ds = interleave_datasets(
                domain_ds_ls,
                probabilities=probabilities,
                probabilities_handle=domain_weight_buffer_handle,
                seed=seed)

    def take_data_generator(ds, max_samples):
        idx = 0
        for ex in ds:
            yield ex
            idx += 1
            if max_samples is not None and idx >= max_samples:
                return

    ds = IterableDataset.from_generator(take_data_generator, gen_kwargs={'ds': ds, 'max_samples': max_samples})
    return ds