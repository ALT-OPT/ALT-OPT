d=Cora; gpu=2; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=ALTOPT0.6; echo ${name}; \
CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
--model ${m}  --prop ${prop} --runs 3 --random_splits 10 --log_steps 100 \
--weight_decay  0.0005 --dropout 0.8  --K 1 --lr 0.01  --proportion 0.6 \
--loop 5 --lambda1 0.05 \
--debug 1 2>&1 | tee -a ./result/${name}_log;

d=PubMed; gpu=0; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=ALTOPT20; echo ${name}; \
CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} \
--model ${m}  --prop ${prop} --runs 3 --random_splits 10 --log_steps 100 \
--weight_decay  0.0005 --dropout 0.8  --K 1 --lr 0.01  --fix_num 20 \
--loop 5 \
--debug 1 2>&1 | tee -a ./result/${name}${d}_log;