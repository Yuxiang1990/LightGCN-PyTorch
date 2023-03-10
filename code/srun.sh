srun_args='-p pat_mercury -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=vessel --kill-on-bad-exit=1'

srun $srun_args python -u -m main --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" \
--topks="[20]" --recdim=64 --model="lgn_v1_fast"
