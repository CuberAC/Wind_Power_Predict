import numpy as np
import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

def main():
    print("🚀 开始加载第一层 OOF 和 Test 预测数据...")
    
    # 1. 加载数据
    try:
        xgb_oof = np.load('data/stacking_data/XGB_OOF_Pred.npy')
        gnn_oof = np.load('data/stacking_data/GNN_OOF_Pred.npy')
        y_oof   = np.load('data/stacking_data/Y_OOF_True.npy')

        xgb_test = np.load('data/stacking_data/XGB_Test_Pred.npy')
        gnn_test = np.load('data/stacking_data/GNN_Test_Pred.npy')
        y_test   = np.load('data/stacking_data/Y_Test_True.npy')
    except FileNotFoundError as e:
        print(f"❌ 找不到文件: {e} (请检查 stacking_data 目录下的文件名是否拼写正确)")
        return

    # 2. 形状核对 (防翻车第一定律)
    print(f"   => XGB OOF 形状: {xgb_oof.shape} | GNN OOF 形状: {gnn_oof.shape}")
    print(f"   => XGB Test 形状: {xgb_test.shape} | GNN Test 形状: {gnn_test.shape}")
    
    if xgb_oof.shape != gnn_oof.shape:
        print("❌ 致命错误：XGB 和 GNN 的 OOF 样本数不一致，请检查切分逻辑！")
        return

    print("\n" + "-"*40)
    print("📊 融合前：单模型在测试集上的真实战力")
    xgb_mae = mean_absolute_error(y_test, xgb_test)
    gnn_mae = mean_absolute_error(y_test, gnn_test)
    print(f"   XGBoost MAE : {xgb_mae:.6f}")
    print(f"   GNN_LSTM MAE: {gnn_mae:.6f}")
    print("-"*40)

    # 3. 构建第二层训练矩阵 (Meta-Features)
    # 【核心】：不加原气象特征！只用预测值。总维度 = 24(XGB) + 24(GNN) = 48维
    print("\n🧠 正在训练第二层元模型 (Ridge Regression)...")
    Meta_X_Train = np.concatenate([xgb_oof, gnn_oof], axis=1)
    Meta_X_Test  = np.concatenate([xgb_test, gnn_test], axis=1)

    # 使用岭回归，alpha=1.0 起到 L2 正则化作用，防止极端权重
    meta_model = MultiOutputRegressor(Ridge(alpha=1.0))
    meta_model.fit(Meta_X_Train, y_oof)

    # 4. 进行终极测试集预测
    Final_Stack_Pred = meta_model.predict(Meta_X_Test)
    stack_mae = mean_absolute_error(y_test, Final_Stack_Pred)
    stack_rmse = np.sqrt(mean_squared_error(y_test, Final_Stack_Pred))

    # 5. 顺手做个极简加权 (Blending Baseline) 作为对照
    print("\n⚖️ 正在寻找最佳简单加权比例...")
    best_w, best_oof_mae = 0, float('inf')
    # 穷举一下，看看 XGB 占多少比例最好
    for w in np.arange(0.5, 1.0, 0.05):
        blend_oof = w * xgb_oof + (1 - w) * gnn_oof
        tmp_mae = mean_absolute_error(y_oof, blend_oof)
        if tmp_mae < best_oof_mae:
            best_oof_mae = tmp_mae
            best_w = w
            
    print(f"   => OOF 选出的黄金比例: {best_w:.2f} * XGB + {(1-best_w):.2f} * GNN")
    Final_Blend_Pred = best_w * xgb_test + (1 - best_w) * gnn_test
    blend_mae = mean_absolute_error(y_test, Final_Blend_Pred)

    # 6. 终极宣判
    print("\n" + "="*50)
    print("🏆 大决战：风电 Stacking 终极成绩单 🏆")
    print(f"   Ridge Stacking MAE : {stack_mae:.6f}  <-- 【核心看这里！】")
    print(f"   最佳加权 Blend MAE : {blend_mae:.6f}")
    print(f"   (作为参考) XGB MAE : {xgb_mae:.6f}")
    print("="*50)

    # 7. 保存元模型
    os.makedirs('saved_models/final_ensemble', exist_ok=True)
    joblib.dump(meta_model, 'saved_models/final_ensemble/layer2_ridge.pkl')
    print("💾 第二层元模型已保存至 saved_models/final_ensemble/layer2_ridge.pkl")

if __name__ == "__main__":
    main()