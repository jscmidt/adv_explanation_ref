# [1] Relu-Like Gradient, config as in paper
python3 run_attack.py --method gradient --num_iter 1500 --lr 1e-3 --beta_growth --output-prefix "1"
# [2] Relu LRP, config as in paper
python3 run_attack.py --method lrp --num_iter 1500 --lr 2e-4 --output-prefix "2"
# [3] Softplus beta=0.8 gradient, config as in paper (weirdly different from Relu-Like Gradient[1])
python3 run_attack.py --method gradient --num_iter 500 --lr 2.5e-4 --beta 0.8 --output-prefix "3"
# [4] Softplus beta=0.8 LRP, config as in paper (and actually the same as Relu-LRP[2])
python3 run_attack.py --method lrp --num_iter 1500 --lr 2e-4 --beta 0.8 --output-prefix "4"
# [5] Softplus beta=0.8 gradient, but with same config as Relu-Like Gradient[1]
python3 run_attack.py --method gradient --num_iter 1500 --lr 1e-3 --beta 0.8 --output-prefix "5"

# For comparison with old config
python3 run_attack.py --method gradient --beta_growth --output-prefix "11"
python3 run_attack.py --method gradient --beta 0.8 --output-prefix "12"
python3 run_attack.py --output-prefix "13"
python3 run_attack.py --beta 0.8 --output-prefix "14"
