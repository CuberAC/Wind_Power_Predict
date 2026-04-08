# make_req.py
import numpy
import pandas
import xgboost
import sklearn
import joblib
import optuna

def generate_requirements():
    # 获取各个库的当前版本号
    reqs = [
        f"numpy=={numpy.__version__}",
        f"pandas=={pandas.__version__}",
        f"xgboost=={xgboost.__version__}",
        f"scikit-learn=={sklearn.__version__}",  # 注意：安装包叫 scikit-learn，代码里叫 sklearn
        f"joblib=={joblib.__version__}",
        f"optuna=={optuna.__version__}" 
    ]
    
    # 写入到 requirements.txt 文件中
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        for req in reqs:
            f.write(req + '\n')
            
    print("✅ 成功生成 requirements.txt 文件！内容如下：")
    print("-" * 30)
    for req in reqs:
        print(req)
    print("-" * 30)

if __name__ == "__main__":
    generate_requirements()