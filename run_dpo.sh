nohup python -u train.py \
    loss=dpo \
    datasets=[eeyore_dpo] \
    lr=5e-7 \
    exp_name=dpo-eeyore \
    n_epochs=1 \
    batch_size=6 \
    eval_batch_size=6 \
    model.archive=/home/20nguyen.hk/dpo-eeyore/.cache/20nguyen.hk/sft-eeyore_2025-07-20_08-07-47_636201/LATEST/policy.pt \
    > dpo.log &