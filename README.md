训练STGCN:
python train_stgcn.py \
  --data-path data/wind_train_val_2012-01-02_to_2013-07-13.npy \
  --window-size 24 \
  --batch-size 64 \
  --epochs 200 \
  --lr 0.005 \
  --hidden-dim 64 \
  --alpha 3.0 \
  --patience 20 \
  --dropout 0.1

