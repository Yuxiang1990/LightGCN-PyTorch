srun_args='-p pat_mercury -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=vessel --kill-on-bad-exit=1'

#['lastfm', 'gowalla', 'yelp2018', 'amazon-book']
#python -u -m main --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="amazon-book" \
#--topks="[20]" --recdim=64 --model="lgn" --comment="light-gcn-amazon-book"

srun $srun_args python -u -m main --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset=$1 \
--topks="[20]" --recdim=64 --model="lgn" --comment="light-gcn-${1}-svd${2}" --svd_num=${2}


# light gcn
#srun $srun_args python -u -m main --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" \
#--topks="[20]" --recdim=64 --model="lgn" --comment="light-gcn-svd90_new" --svd_num=90


# light gcn
#srun $srun_args python -u -m main --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" \
#--topks="[20]" --recdim=64 --model="lgn" --comment="light-gcn-svd60" --svd_num=60


# light gcn
#srun $srun_args python -u -m main --decay=1e-4 --lr=0.01 --layer=3 --seed=2020 --dataset="gowalla" \
#--topks="[20]" --recdim=64 --model="lgn" --comment="light-gcn-svd180" --svd_num=180



# light-gcn svd
#alpha=3, req_vec=90, beta=2.0, coef_u=0.5, coef_i=0.9
#learning_rate: b(7.5) u (10.5) i  (10.5) m(9.0)
#srun $srun_args python -u -m main --decay=0.01 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" \
#--topks="[20]" --recdim=64 --model="lgn_svd" --comment="req_vec90_decay0.1_bs256" --epochs=100 --bpr_batch=256

#python -u -m main --decay=0.01 --lr=0.1 --layer=3 --seed=2020 --dataset="gowalla" \
#--topks="[20]" --recdim=64 --model="lgn_svd" --comment="req_vec90_decay0.1_bs256" --epochs=100 --bpr_batch=256
