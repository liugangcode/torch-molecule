python examples/predict_GREA_for_gas.py --data_path private/gas_permeability.csv --task CO2 --n_trials 2
 python examples/train_GNN_for_any.py --data_path private/tg.csv --model_type GNN
 python examples/train_GNN_for_any.py --data_path private/tg.csv --model_type GREA

python examples/predict_GREA.py --data_path private/polyinfo.csv --log_scale
python examples/predict_GREA.py --data_path private/gas_permeability.csv --log_scale
