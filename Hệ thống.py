import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r'F:\BTL\Phân tích dữ liệu\winequality-red.csv', sep=';')

# 1. Tổng quan
print("Kích thước:", df.shape)
print(df.dtypes)
print("Giá trị thiếu:\n", df.isnull().sum())
print(df.describe().T)

# 2. Phân phối biến mục tiêu
print("\nPhân phối quality:\n", df['quality'].value_counts().sort_index())

# 3. Tương quan với quality
corr = df.corr()['quality'].drop('quality').sort_values(key=abs, ascending=False)
print("\nTương quan Pearson với quality:\n", corr)

# 4. Outlier theo IQR
print("\nSố outlier theo IQR:")
for col in df.columns.drop('quality'):
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"{col}: {n_out}")

# 5. Mô hình dự đoán
X = df.drop(columns='quality')
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
pred_lr = lr.predict(X_test)
print("\nLinear Regression R2:", round(r2_score(y_test, pred_lr), 4))
print("Linear Regression RMSE:", round(np.sqrt(mean_squared_error(y_test, pred_lr)), 4))

rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train, y_train)
pred_rf = rf.predict(X_test)
print("\nRandom Forest R2:", round(r2_score(y_test, pred_rf), 4))
print("Random Forest RMSE:", round(np.sqrt(mean_squared_error(y_test, pred_rf)), 4))

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importance (Random Forest):\n", importances)

# 6. Hồi quy tuyến tính đa biến đầy đủ (statsmodels OLS)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_const = sm.add_constant(X)
ols_model = sm.OLS(y, X_const).fit()
print("\n" + str(ols_model.summary()))

vif = pd.DataFrame()
vif['variable'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVIF:\n", vif.sort_values('VIF', ascending=False))

# 7. Biểu đồ
import matplotlib.pyplot as plt

cols = df.columns.drop('quality')

# 7.1 Histogram phân phối từng biến
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(cols):
    axes[i].hist(df[col], bins=30, color='#4C72B0', edgecolor='black')
    axes[i].set_title(col, fontsize=10)
axes[-1].hist(df['quality'], bins=range(3, 10), color='#DD8452', edgecolor='black')
axes[-1].set_title('quality', fontsize=10)
plt.tight_layout()
plt.savefig('histograms.png', dpi=120)
print("Đã lưu:", os.path.abspath('histograms.png'))
plt.show()
plt.close()

# 7.2 Boxplot phát hiện outlier từng biến
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(cols):
    axes[i].boxplot(df[col], orientation='vertical')
    axes[i].set_title(col, fontsize=10)
axes[-1].axis('off')
plt.tight_layout()
plt.savefig('boxplots.png', dpi=120)
print("Đã lưu:", os.path.abspath('boxplots.png'))
plt.show()
plt.close()

# 7.3 Heatmap ma trận tương quan
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_yticklabels(corr_matrix.columns, fontsize=8)
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha='center', va='center', fontsize=6)
plt.colorbar(im)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=120)
print("Đã lưu:", os.path.abspath('correlation_heatmap.png'))
plt.show()
plt.close()

# 7.4 Bar chart hệ số hồi quy OLS
ols_coefs = ols_model.params.drop('const')
plt.figure(figsize=(10, 6))
ols_coefs.sort_values().plot(kind='barh', color='#4C72B0')
plt.title('Hệ số hồi quy tuyến tính đa biến (OLS)')
plt.xlabel('Hệ số')
plt.tight_layout()
plt.savefig('ols_coefficients.png', dpi=120)
print("Đã lưu:", os.path.abspath('ols_coefficients.png'))
plt.show()
plt.close()

# 7.5 Bar chart tương quan Pearson với quality
plt.figure(figsize=(10, 6))
corr.sort_values().plot(kind='barh', color='#DD8452')
plt.title('Tương quan Pearson với quality')
plt.xlabel('Hệ số tương quan')
plt.tight_layout()
plt.savefig('correlation_with_quality.png', dpi=120)
print("Đã lưu:", os.path.abspath('correlation_with_quality.png'))
plt.show()
plt.close()

# ============================================================
# 8. ACTUAL VS PREDICTED
# ============================================================
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, pred_lr)
mse = mean_squared_error(y_test, pred_lr)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred_lr)
print(f"\nMAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred_lr, alpha=0.4, color='#4C72B0')
lims = [min(y_test.min(), pred_lr.min()), max(y_test.max(), pred_lr.max())]
plt.plot(lims, lims, 'r--', label='y = x (lý tưởng)')
plt.title('Actual vs Predicted')
plt.xlabel('Giá trị thực tế'); plt.ylabel('Giá trị dự đoán')
plt.legend(); plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=120)
print("Đã lưu:", os.path.abspath('actual_vs_predicted.png'))
plt.show()
plt.close()

# ============================================================
# 9. PHÂN TÍCH RESIDUAL / SAI SỐ
# ============================================================
import scipy.stats as stats

residuals = y_test.values - pred_lr

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].scatter(pred_lr, residuals, alpha=0.4, color='#4C72B0')
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_title('Residual vs Predicted')
axes[0].set_xlabel('Giá trị dự đoán'); axes[0].set_ylabel('Residual')

axes[1].hist(residuals, bins=30, color='#DD8452', edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_title('Phân phối Residual')
axes[1].set_xlabel('Residual'); axes[1].set_ylabel('Số mẫu')

stats.probplot(residuals, dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot Residual')

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=120)
print("Đã lưu:", os.path.abspath('residual_analysis.png'))
plt.show()
plt.close()

jb_stat, jb_p = stats.jarque_bera(residuals)[:2]
print(f"\nJarque-Bera (chuẩn tắc residual): stat={jb_stat:.4f}  p={jb_p:.6f}")
print(f"Sai số tuyệt đối trung bình: {np.mean(np.abs(residuals)):.4f}")
print(f"Sai số lớn nhất: {np.max(np.abs(residuals)):.4f}")

# ============================================================
# 10. PHÂN TÍCH HỆ SỐ HỒI QUY (COEFFICIENTS)
# ============================================================
pvalues = ols_model.pvalues.drop('const')
sig_vars = pvalues[pvalues < 0.05].index.tolist()
print("\nBiến có ý nghĩa thống kê (p<0.05):", sig_vars)

# ============================================================
# 11. HỆ SỐ CHUẨN HÓA (STANDARDIZED COEFFICIENTS)
# ============================================================
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
X_scaled = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns)

X_scaled_const = sm.add_constant(X_scaled)
ols_std = sm.OLS(y, X_scaled_const).fit()
std_coefs = ols_std.params.drop('const').sort_values(key=abs, ascending=False)

print("\nHệ số chuẩn hóa (sắp theo |beta| giảm dần):")
print(std_coefs)

plt.figure(figsize=(9, 6))
colors_std = ['#C44E52' if c < 0 else '#55A868' for c in std_coefs.sort_values().values]
std_coefs.sort_values().plot(kind='barh', color=colors_std, edgecolor='black')
plt.title('Hệ số hồi quy chuẩn hóa (Standardized Coefficients / Beta)')
plt.xlabel('Beta (độ lệch chuẩn)')
plt.tight_layout()
plt.savefig('standardized_coefficients.png', dpi=120)
print("Đã lưu:", os.path.abspath('standardized_coefficients.png'))
plt.show()
plt.close()

print(f"\nBiến có ảnh hưởng mạnh nhất (theo |beta|): {std_coefs.index[0]} (beta={std_coefs.iloc[0]:.4f})")