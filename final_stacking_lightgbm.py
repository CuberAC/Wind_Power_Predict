import numpy as np
import os
import joblib
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor


def main():
    print("🚀 开始加载第一层 OOF 和 Test 预测数据...")

    # 1. 加载数据
    try:
        xgb_oof = np.load("stacking_data/XGB_OOF_Pred.npy")
        gnn_oof = np.load("stacking_data/GNN_OOF_Pred.npy")
        y_oof = np.load("stacking_data/Y_OOF_True.npy")

        xgb_test = np.load("stacking_data/XGB_Test_Pred.npy")
        gnn_test = np.load("stacking_data/GNN_Test_Pred.npy")
        y_test = np.load("stacking_data/Y_Test_True.npy")
    except FileNotFoundError as e:
        print(f"❌ 找不到文件: {e} (请检查 stacking_data 目录下的文件名是否拼写正确)")
        return

    # 2. 形状核对
    print(f"   => XGB OOF 形状: {xgb_oof.shape} | GNN OOF 形状: {gnn_oof.shape}")
    print(f"   => XGB Test 形状: {xgb_test.shape} | GNN Test 形状: {gnn_test.shape}")

    if xgb_oof.shape != gnn_oof.shape:
        print("❌ 致命错误：XGB 和 GNN 的 OOF 样本数不一致，请检查切分逻辑！")
        return

    print("\n" + "-" * 40)
    print("📊 融合前：单模型在测试集上的真实战力")
    xgb_mae = mean_absolute_error(y_test, xgb_test)
    gnn_mae = mean_absolute_error(y_test, gnn_test)
    print(f"   XGBoost MAE : {xgb_mae:.6f}")
    print(f"   GNN_LSTM MAE: {gnn_mae:.6f}")
    print("-" * 40)

    # 3. 构建第二层训练矩阵 (Meta-Features)
    print("\n🧠 正在训练第二层元模型 (LightGBM)...")
    meta_x_train = np.concatenate([xgb_oof, gnn_oof], axis=1)
    meta_x_test = np.concatenate([xgb_test, gnn_test], axis=1)

    # 使用极度受限的 LightGBM 防止元过拟合
    meta_model = MultiOutputRegressor(
        lgb.LGBMRegressor(
            n_estimators=1000,
            max_depth=1,
            num_leaves=3,
            learning_rate=0.01,
            min_child_samples=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        )
    )
    meta_model.fit(meta_x_train, y_oof)

    # 4. 进行终极测试集预测
    final_stack_pred = meta_model.predict(meta_x_test)
    stack_mae = mean_absolute_error(y_test, final_stack_pred)
    stack_rmse = np.sqrt(mean_squared_error(y_test, final_stack_pred))

    # 5. 简单加权对照
    print("\n⚖️ 正在寻找最佳简单加权比例...")
    best_w, best_oof_mae = 0, float("inf")
    for w in np.arange(0.5, 1.0, 0.05):
        blend_oof = w * xgb_oof + (1 - w) * gnn_oof
        tmp_mae = mean_absolute_error(y_oof, blend_oof)
        if tmp_mae < best_oof_mae:
            best_oof_mae = tmp_mae
            best_w = w

    print(f"   => OOF 选出的黄金比例: {best_w:.2f} * XGB + {(1 - best_w):.2f} * GNN")
    final_blend_pred = best_w * xgb_test + (1 - best_w) * gnn_test
    blend_mae = mean_absolute_error(y_test, final_blend_pred)

    # 6. 终极成绩
    print("\n" + "=" * 50)
    print("🏆 大决战：风电 Stacking 终极成绩单 🏆")
    print(f"   LightGBM Stacking MAE : {stack_mae:.6f}  <-- 【核心看这里！】")
    print(f"   LightGBM Stacking RMSE: {stack_rmse:.6f}")
    print(f"   最佳加权 Blend MAE    : {blend_mae:.6f}")
    print(f"   (作为参考) XGB MAE    : {xgb_mae:.6f}")
    print("=" * 50)

    # 7. 保存元模型
    os.makedirs("saved_models/final_ensemble", exist_ok=True)
    save_path = "saved_models/final_ensemble/layer2_lightgbm.pkl"
    joblib.dump(meta_model, save_path)
    print(f"💾 第二层元模型已保存至 {save_path}")


if __name__ == "__main__":
    main()
