
d=CiteSeer; gpu=1; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT10_CiteSeer; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 10 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 \
 --debug 1 2>&1 | tee -a ./result/${name}_log;

d=CiteSeer; gpu=1; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT5_CiteSeer; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 5 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 2>&1 | tee -a ./result/${name}_log;

d=CiteSeer; gpu=1; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT60_CiteSeer; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 60 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 2>&1 | tee -a ./result/${name}_log;


#Label propagation:
#d=Cora; gpu=3; h=64; echo hidden ${h};
#m=ALTOPT; prop=APPNP; name=test; echo ${name}; \
# CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
# --model ${m}  --prop ${prop} --runs 3 --random_splits 0 --lr 0.01 --dropout 0.5 --K 10 \
# --log_steps 100 --weight_decay 0.0005 --sort_key lambda1 --lambda1 0  --lambda2 0 --alpha 0.1 --hidden_channels ${h} --LP True \
# --debug 1 2>&1 | tee -a ./result/${name}_log;
#
#
#APPNP:
#d=Cora; gpu=3; h=64; echo hidden ${h};
#m=APPNP; prop=APPNP; name=test; echo ${name}; \
# CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
# --model ${m}  --prop ${prop} --runs 3 --random_splits 0 --lr 0.01 --dropout 0.5 --K 10 \
# --log_steps 100 --weight_decay 0.0005 --sort_key lambda1 --lambda1 0  --lambda2 0 --alpha 0.1 --hidden_channels ${h} \
# --debug 1 2>&1 | tee -a ./result/${name}_log;
#
#
#
#d=Cora; gpu=3; h=64; echo hidden ${h};
#m=ALTOPT; prop=APPNP; name=test; echo ${name}; \
# CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
# --model ${m}  --prop ${prop} --runs 3 --random_splits 0 --lr 0.01 --dropout 0.5 --K 10 \
# --log_steps 100 --weight_decay 0.0005 --sort_key lambda1 --lambda1 0  --lambda2 0 --alpha 0.1 --hidden_channels ${h} \
# --debug 1 2>&1 | tee -a ./result/${name}_log;
#
#
#d=Cora; gpu=3; h=64; echo hidden ${h};
#m=ALTOPT; prop=ALTOPT; name=test; echo ${name}; \
# CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
# --model ${m}  --prop ${prop} --runs 3 --random_splits 0 --lr 0.01 --dropout 0.5 --K 10 \
# --log_steps 100 --weight_decay 0.0005 --sort_key lambda1 --lambda1 0  --lambda2 0 --alpha 0.1 --hidden_channels ${h} \
# --debug 1 2>&1 | tee -a ./result/${name}_log;
#
#
#search lambda
# d=Cora; gpu=3; h=64; echo hidden ${h};
#m=ALTOPT; prop=ALTOPT; name=test; echo ${name}; \
# CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
# --model ${m}  --prop ${prop} --runs 2 --random_splits 0 --lr 0.01 --dropout 0.5 --K 10 \
# --log_steps 200 --weight_decay 0.0005 --sort_key lambda1 --alpha 0.1 --hidden_channels ${h} \
# --debug 1 2>&1 | tee -a ./result/${name}_log;