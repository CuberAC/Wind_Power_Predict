运行XGBoost的训练：
python train_xgb.py --data features_v1.npz

xgb_v2搜索出来的最佳参数：
    n_estimators=598,
    max_depth=7,
    learning_rate=0.013698166894754917,
    subsample=0.7186255612121745,
    colsample_bytree=0.9993604825235307,
    min_child_weight=2,
    objective='reg:squarederror',
    n_jobs=-1,
    random_state=42

xgb_v3搜索出来的最佳参数：
    n_estimators: 1230
    max_depth: 9
    learning_rate: 0.010659217252788872
    subsample: 0.6523662083125326
    colsample_bytree: 0.7510208285575809
    min_child_weight: 5

LSTM模型的训练：
python LSTM_baseline.py \
  --epochs 200 \
  --batch-size 256 \
  --lr 0.001 \
  --hidden-dim 128 \
  --num-layers 3 \
  --save-every 20
  
LSTM模型的评测：
python evalutate_lstm.py logs/20260409_151738/lstm_epoch_100.pth