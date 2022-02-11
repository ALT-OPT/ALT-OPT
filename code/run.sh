APPNP:
d=PubMed; gpu=0; h=64; echo hidden ${h};
m=APPNP; prop=APPNP; name=APPNP20; echo ${name}; \
 CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
 --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --fix_num 20 \
 --lr 0.01 --K 10 \
 --log_steps 100 --hidden_channels ${h} --weight_decay 0.0005\
 --debug 1 2>&1 | tee -a ./result/${name}_log;

# APPNP:
#d=Cora; gpu=3; h=64; echo hidden ${h};
#m=APPNP; prop=APPNP; name=APPNP140; echo ${name}; \
# CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
# --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --fix_num 140 \
# --lr 0.01 --K 10 \
# --log_steps 100 --hidden_channels ${h} --weight_decay 0.0005\
# --debug 1 2>&1 | tee -a ./result/${name}_log;

# APPNP:
#d=Cora; gpu=3; h=64; echo hidden ${h};
#m=APPNP; prop=APPNP; name=APPNP60; echo ${name}; \
# CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
# --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --fix_num 60 \
# --lr 0.01 --K 10 \
# --log_steps 100 --hidden_channels ${h} --weight_decay 0.0005\
# --debug 1 2>&1 | tee -a ./result/${name}_log;
#
# APPNP:
#d=Cora; gpu=3; h=64; echo hidden ${h};
#m=APPNP; prop=APPNP; name=APPNP80; echo ${name}; \
# CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
# --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --fix_num 80 \
# --lr 0.01 --K 10 \
# --log_steps 100 --hidden_channels ${h} --weight_decay 0.0005\
# --debug 1 2>&1 | tee -a ./result/${name}_log;
#
# APPNP:
#d=Cora; gpu=3; h=64; echo hidden ${h};
#m=APPNP; prop=APPNP; name=APPNP100; echo ${name}; \
# CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
# --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --fix_num 100 \
# --lr 0.01 --K 10 \
# --log_steps 100 --hidden_channels ${h} --weight_decay 0.0005\
# --debug 1 2>&1 | tee -a ./result/${name}_log;
#
# APPNP:
#d=Cora; gpu=3; h=64; echo hidden ${h};
#m=APPNP; prop=APPNP; name=APPNP0.6; echo ${name}; \
# CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
# --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --proportion 0.6 \
# --lr 0.01 --K 10 \
# --log_steps 100 --hidden_channels ${h} --weight_decay 0.0005\
# --debug 1 2>&1 | tee -a ./result/${name}_log;