
d=CiteSeer; gpu=2; h=64; echo hidden ${h};
m=LP; prop=LP; name=LP5_CiteSeer; echo ${name};  CUDA_VISIBLE_DEVICES=${gpu};
python3 -u main_optuna.py --dataset ${d}  --model ${m}  --prop ${prop} \
--runs 1 --random_splits 10  --fix_num 5 --lr 0.01 --weight_decay 0.0005 \
--dropout 0.8 --log_steps 100 --hidden_channels ${h}  --K 10 \
--debug 1 2>&1 | tee -a ./result/${name}_log;

d=CiteSeer; gpu=2; h=64; echo hidden ${h};
m=LP; prop=LP; name=LP10_CiteSeer; echo ${name};  CUDA_VISIBLE_DEVICES=${gpu};
python3 -u main_optuna.py --dataset ${d}  --model ${m}  --prop ${prop} \
--runs 1 --random_splits 10  --fix_num 10 --lr 0.01 --weight_decay 0.0005 \
--dropout 0.8 --log_steps 100 --hidden_channels ${h}  --K 10 \
--debug 1 2>&1 | tee -a ./result/${name}_log;

d=CiteSeer; gpu=2; h=64; echo hidden ${h};
m=LP; prop=LP; name=LP20_CiteSeer; echo ${name};  CUDA_VISIBLE_DEVICES=${gpu};
python3 -u main_optuna.py --dataset ${d}  --model ${m}  --prop ${prop} \
--runs 1 --random_splits 10  --fix_num 20 --lr 0.01 --weight_decay 0.0005 \
--dropout 0.8 --log_steps 100 --hidden_channels ${h}  --K 10 \
--debug 1 2>&1 | tee -a ./result/${name}_log;

d=CiteSeer; gpu=2; h=64; echo hidden ${h};
m=LP; prop=LP; name=LP60_CiteSeer; echo ${name};  CUDA_VISIBLE_DEVICES=${gpu};
python3 -u main_optuna.py --dataset ${d}  --model ${m}  --prop ${prop} \
--runs 1 --random_splits 10  --fix_num 60 --lr 0.01 --weight_decay 0.0005 \
--dropout 0.8 --log_steps 100 --hidden_channels ${h}  --K 10 \
--debug 1 2>&1 | tee -a ./result/${name}_log;

d=CiteSeer; gpu=2; h=64; echo hidden ${h};
m=LP; prop=LP; name=LP0.3_CiteSeer; echo ${name};  CUDA_VISIBLE_DEVICES=${gpu};
python3 -u main_optuna.py --dataset ${d}  --model ${m}  --prop ${prop} \
--runs 1 --random_splits 10  --proportion 0.3 --lr 0.01 --weight_decay 0.0005 \
--dropout 0.8 --log_steps 100 --hidden_channels ${h}  --K 10 \
--debug 1 2>&1 | tee -a ./result/${name}_log;

d=CiteSeer; gpu=2; h=64; echo hidden ${h};
m=LP; prop=LP; name=LP0.6_CiteSeer; echo ${name};  CUDA_VISIBLE_DEVICES=${gpu};
python3 -u main_optuna.py --dataset ${d}  --model ${m}  --prop ${prop} \
--runs 1 --random_splits 10  --proportion 0.6 --lr 0.01 --weight_decay 0.0005 \
--dropout 0.8 --log_steps 100 --hidden_channels ${h}  --K 10 \
--debug 1 2>&1 | tee -a ./result/${name}_log;