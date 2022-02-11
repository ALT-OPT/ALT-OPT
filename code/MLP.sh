d=CiteSeer; gpu=2; h=64; echo hidden ${h};
m=MLP; prop=MLP; name=MLP5_CiteSeer; echo ${name}; \
 CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
 --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --fix_num 5 \
 --log_steps 100 --hidden_channels ${h} --lr 0.01 --alpha 0 --K 0 --pro_alpha 0\
 --weight_decay 0.0005 --debug 1 2>&1 | tee -a ./result/${name}_log;

 d=CiteSeer; gpu=2; h=64; echo hidden ${h};
m=MLP; prop=MLP; name=MLP10_CiteSeer; echo ${name}; \
 CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
 --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --fix_num 10 \
 --log_steps 100 --hidden_channels ${h} --lr 0.01 --alpha 0 --K 0 --pro_alpha 0\
 --weight_decay 0.0005 --debug 1 2>&1 | tee -a ./result/${name}_log;

d=CiteSeer; gpu=2; h=64; echo hidden ${h};
m=MLP; prop=MLP; name=MLP20_CiteSeer; echo ${name}; \
 CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
 --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --fix_num 20 \
 --log_steps 100 --hidden_channels ${h} --lr 0.01 --alpha 0 --K 0 --pro_alpha 0\
 --weight_decay 0.0005 --debug 1 2>&1 | tee -a ./result/${name}_log;

 d=CiteSeer; gpu=2; h=64; echo hidden ${h};
m=MLP; prop=MLP; name=MLP60_CiteSeer; echo ${name}; \
 CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
 --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --fix_num 60 \
 --log_steps 100 --hidden_channels ${h} --lr 0.01 --alpha 0 --K 0 --pro_alpha 0\
 --weight_decay 0.0005 --debug 1 2>&1 | tee -a ./result/${name}_log;

  d=CiteSeer; gpu=2; h=64; echo hidden ${h};
m=MLP; prop=MLP; name=MLP0.3_CiteSeer; echo ${name}; \
 CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
 --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --proportion 0.3 \
 --log_steps 100 --hidden_channels ${h} --lr 0.01 --alpha 0 --K 0 --pro_alpha 0\
 --weight_decay 0.0005 --debug 1 2>&1 | tee -a ./result/${name}_log;

 d=CiteSeer; gpu=2; h=64; echo hidden ${h};
m=MLP; prop=MLP; name=MLP0.6_CiteSeer; echo ${name}; \
 CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
 --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --proportion 0.6 \
 --log_steps 100 --hidden_channels ${h} --lr 0.01 --alpha 0 --K 0 --pro_alpha 0\
 --weight_decay 0.0005 --debug 1 2>&1 | tee -a ./result/${name}_log;

# d=Cora; gpu=1; h=64; echo hidden ${h};
#m=MLP; prop=MLP; name=MLP60; echo ${name}; \
# CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
# --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --fix_num 60 \
# --log_steps 100 --hidden_channels ${h}\
# --debug 1 2>&1 | tee -a ./result/${name}_log;
#
# d=Cora; gpu=1; h=64; echo hidden ${h};
#m=MLP; prop=MLP; name=MLP80; echo ${name}; \
# CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
# --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --fix_num 80 \
# --log_steps 100 --hidden_channels ${h}\
# --debug 1 2>&1 | tee -a ./result/${name}_log;
#
# d=Cora; gpu=1; h=64; echo hidden ${h};
#m=MLP; prop=MLP; name=MLP100; echo ${name}; \
# CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
# --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --fix_num 100 \
# --log_steps 100 --hidden_channels ${h}\
# --debug 1 2>&1 | tee -a ./result/${name}_log;
#
# d=Cora; gpu=1; h=64; echo hidden ${h};
#m=MLP; prop=MLP; name=MLP0.6; echo ${name}; \
# CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
# --model ${m}  --prop ${prop} --runs 3 --random_splits 10  --proportion 0.6 \
# --log_steps 100 --hidden_channels ${h}\
# --debug 1 2>&1 | tee -a ./result/${name}_log;