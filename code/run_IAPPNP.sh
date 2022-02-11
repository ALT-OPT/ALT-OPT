d=CiteSeer; gpu=0; h=64; echo hidden ${h};
m=IAPPNP; prop=APPNP; name=IAPPNP5_CiteSeer; echo ${name};  CUDA_VISIBLE_DEVICES=${gpu};
python3 -u main_optuna.py --dataset ${d}  --model ${m}  --prop ${prop} --runs 3 --random_splits 10 --log_steps 100 \
--fix_num 5  --lr 0.01 --K 10 --weight_decay 0.0005 --hidden_channels ${h} --pro_alpha 0 \
--debug 1 2>&1 | tee -a ./result/${name}_log;

d=CiteSeer; gpu=0; h=64; echo hidden ${h};
m=IAPPNP; prop=APPNP; name=IAPPNP10_CiteSeer; echo ${name};  CUDA_VISIBLE_DEVICES=${gpu};
python3 -u main_optuna.py --dataset ${d}  --model ${m}  --prop ${prop} --runs 3 --random_splits 10 --log_steps 100 \
--fix_num 10  --lr 0.01 --K 10 --weight_decay 0.0005 --hidden_channels ${h} --pro_alpha 0 \
--debug 1 2>&1 | tee -a ./result/${name}_log;

d=CiteSeer; gpu=0; h=64; echo hidden ${h};
m=IAPPNP; prop=APPNP; name=IAPPNP20_CiteSeer; echo ${name};  CUDA_VISIBLE_DEVICES=${gpu};
python3 -u main_optuna.py --dataset ${d}  --model ${m}  --prop ${prop} --runs 3 --random_splits 10 --log_steps 100 \
--fix_num 20  --lr 0.01 --K 10 --weight_decay 0.0005 --hidden_channels ${h} --pro_alpha 0 \
--debug 1 2>&1 | tee -a ./result/${name}_log;