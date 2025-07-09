from preference_datasets import get_batch_iterator
import hydra
from omegaconf import OmegaConf, DictConfig
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
import os
from typing import Optional, Set
import transformers
OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
# def main():
    # #Unit test 
        # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)
    
    tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
    train_iterator = get_batch_iterator(names=config.datasets, tokenizer=tokenizer, shuffle=True, max_length=config.max_length, max_prompt_length=config.max_prompt_length, sft_mode=config.loss.name == 'sft', split='train', n_epochs=1, n_examples=1, batch_size=1, silent=True, cache_dir=get_local_dir(config.local_dirs))
    
    print(train_iterator)  
    for batch in train_iterator:
        print(batch)
        

if __name__ == '__main__':
    main()