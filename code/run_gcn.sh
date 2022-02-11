d=CiteSeer; gpu=2; h=64; echo hidden ${h};
m=GCN; prop=GCN; name=GCN5_CireSeer; echo ${name}; \
 CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
 --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --fix_num 5 \
 --log_steps 100 --hidden_channels ${h} --lr 0.01 \
 --debug 1 2>&1 | tee -a ./result/${name}_log;

d=CiteSeer; gpu=2; h=64; echo hidden ${h};
m=GCN; prop=GCN; name=GCN10_CireSeer; echo ${name}; \
 CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
 --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --fix_num 10 \
 --log_steps 100 --hidden_channels ${h} --lr 0.01 \
 --debug 1 2>&1 | tee -a ./result/${name}_log;

d=CiteSeer; gpu=2; h=64; echo hidden ${h};
m=GCN; prop=GCN; name=GCN20_CireSeer; echo ${name}; \
 CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
 --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --fix_num 20 \
 --log_steps 100 --hidden_channels ${h} --lr 0.01 \
 --debug 1 2>&1 | tee -a ./result/${name}_log;

d=CiteSeer; gpu=2; h=64; echo hidden ${h};
m=GCN; prop=GCN; name=GCN60_CireSeer; echo ${name}; \
 CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
 --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --fix_num 60 \
 --log_steps 100 --hidden_channels ${h} --lr 0.01 \
 --debug 1 2>&1 | tee -a ./result/${name}_log;

 d=CiteSeer; gpu=2; h=64; echo hidden ${h};
m=GCN; prop=GCN; name=GCN0.3_CireSeer; echo ${name}; \
 CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
 --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --proportion 0.3 \
 --log_steps 100 --hidden_channels ${h} --lr 0.01 \
 --debug 1 2>&1 | tee -a ./result/${name}_log;

  d=CiteSeer; gpu=2; h=64; echo hidden ${h};
m=GCN; prop=GCN; name=GCN0.6_CireSeer; echo ${name}; \
 CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
 --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --proportion 0.6 \
 --log_steps 100 --hidden_channels ${h} --lr 0.01 \
 --debug 1 2>&1 | tee -a ./result/${name}_log;

# d=Cora; gpu=2; h=64; echo hidden ${h};
#m=GCN; prop=GCN; name=GCN60; echo ${name}; \
# CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
# --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --fix_num 60 \
# --log_steps 100 --hidden_channels ${h} --dropout 0.8 \
# --debug 1 2>&1 | tee -a ./result/${name}_log;
#
# d=Cora; gpu=2; h=64; echo hidden ${h};
#m=GCN; prop=GCN; name=GCN80; echo ${name}; \
# CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
# --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --fix_num 80 \
# --log_steps 100 --hidden_channels ${h} --dropout 0.8 \
# --debug 1 2>&1 | tee -a ./result/${name}_log;
#
# d=Cora; gpu=2; h=64; echo hidden ${h};
#m=GCN; prop=GCN; name=GCN100; echo ${name}; \
# CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
# --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --fix_num 100 \
# --log_steps 100 --hidden_channels ${h} --dropout 0.8 \
# --debug 1 2>&1 | tee -a ./result/${name}_log;
#
# d=Cora; gpu=2; h=64; echo hidden ${h};
#m=GCN; prop=GCN; name=GCN0.6; echo ${name}; \
# CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
# --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --proportion 0.6 \
# --log_steps 100 --hidden_channels ${h} --dropout 0.8 \
# --debug 1 2>&1 | tee -a ./result/${name}_log;