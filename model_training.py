import numpy as np, pandas as pd
#fast maths and for working with tables like in Excel. a Dataframe
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Loading the data
#the model must see clues so it does not cheat
df = pd.read_csv("StudentPerformanceFactors_Cleaned__1_.csv")
X, y = df.drop(columns="Exam_Score"), df["Exam_Score"]

#Splitting into train, validation and test
# 60% train / 20% validation / 20% test
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=42)
print("Train/Val/Test:", len(X_train), len(X_val), len(X_test))

#We are defining the two models
models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8,
                                           min_samples_leaf=5, random_state=42, n_jobs=-1),
}

#This is the scoring fuction part
#its the fuction that we ask to predict the exam score of every student and it uses those three metrics and retuns all three together.
def scores(m, X_, y_):
    p = m.predict(X_)
    return {"MAE": mean_absolute_error(y_, p),
            "RMSE": np.sqrt(mean_squared_error(y_, p)),
            "R2": r2_score(y_, p)}

#Training and validation = so basically what happens here is that that this is our main loop
rows = [] # create an empty list where we collect results
for name, m in models.items():
    m.fit(X_train, y_train) # training step. the model studies the number of students
    for split, (Xs, ys) in {"Train": (X_train, y_train), "Validation": (X_val, y_val)}.items():
        rows.append({"Model": name, "Split": split, **scores(m, Xs, ys)}) #unpacks the metrics
    cv = cross_val_score(m, X_tmp, y_tmp, cv=KFold(5, shuffle=True, random_state=42), scoring="r2") #use 5 folds #shuffle them #random_state=42 make shuffle repeatable
    rows.append({"Model": name, "Split": "5-Fold CV (R2)", "MAE": np.nan, "RMSE": np.nan, "R2": cv.mean()}) #'np.nan menas not a number'


#showing the results table
res = pd.DataFrame(rows).round(3)
print(res.to_string(index=False)) #print the whole table without the extra row-numbers column on the left

# Final test, once, on the chosen model + the comparison model
print("\nFINAL TEST")
for name, m in models.items():
    print(name, {k: round(v, 3) for k, v in scores(m, X_test, y_test).items()})


#Looking inside the models
lr = models["Linear Regression"].named_steps["linearregression"]
print("\nLR standardized coefficients:")
print(pd.Series(lr.coef_, index=X.columns).sort_values(ascending=False).round(3))
print("\nRF importances:")
print(pd.Series(models["Random Forest"].feature_importances_, index=X.columns).sort_values(ascending=False).round(3))









#TESTING PHASE FOR GRAPHS
import matplotlib.pyplot as plt
 
g_names = list(models.keys())                      # ["Linear Regression", "Random Forest"]
g_colors = ["#2E74B5", "#ED7D31"]                  # blue = Linear Regression, orange = Random Forest
g_test = {n: scores(models[n], X_test, y_test) for n in g_names}   # final test metrics
g_pred = {n: models[n].predict(X_test) for n in g_names}           # test predictions
 
# ---------- GRAPH 1: Model comparison on the TEST set (MAE, RMSE, R2) ----------
g_fig, g_axes = plt.subplots(1, 3, figsize=(13, 4.5))
for g_ax, g_metric, g_note in zip(g_axes, ["MAE", "RMSE", "R2"],
                                  ["lower is better", "lower is better", "higher is better"]):
    g_vals = [g_test[n][g_metric] for n in g_names]
    g_bars = g_ax.bar(g_names, g_vals, color=g_colors)
    g_ax.set_title(f"{g_metric} ({g_note})")
    g_ax.bar_label(g_bars, fmt="%.3f", padding=3)
    g_ax.set_ylim(0, max(g_vals) * 1.2)
plt.suptitle("Final test results: Linear Regression vs Random Forest", fontsize=14)
plt.tight_layout()
plt.savefig("graph1_model_comparison.png", dpi=150)
plt.show()
 
# ---------- GRAPH 2: Actual vs Predicted exam scores (test set) ----------
g_fig, g_axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
g_lo, g_hi = y_test.min(), y_test.max()
for g_ax, n, g_col in zip(g_axes, g_names, g_colors):
    g_ax.scatter(y_test, g_pred[n], alpha=0.35, s=14, color=g_col)
    g_ax.plot([g_lo, g_hi], [g_lo, g_hi], "k--", label="Perfect prediction")
    g_ax.set_title(f"{n}  (R2 = {g_test[n]['R2']:.3f})")
    g_ax.set_xlabel("Actual Exam_Score")
    g_ax.set_ylabel("Predicted Exam_Score")
    g_ax.legend()
plt.suptitle("Actual vs Predicted (points closer to the dashed line = better)", fontsize=14)
plt.tight_layout()
plt.savefig("graph2_actual_vs_predicted.png", dpi=150)
plt.show()
 

 
# ---------- GRAPH 4: Feature importance (both models) ----------
g_lr_coef = pd.Series(lr.coef_, index=X.columns).sort_values()
g_rf_imp = pd.Series(models["Random Forest"].feature_importances_, index=X.columns).sort_values()
g_fig, g_axes = plt.subplots(1, 2, figsize=(13, 5))
g_axes[0].barh(g_lr_coef.index, g_lr_coef.values, color=g_colors[0])
g_axes[0].set_title("Linear Regression: scaled coefficients")
g_axes[0].set_xlabel("Points added to predicted score per 1 std. dev.")
g_axes[1].barh(g_rf_imp.index, g_rf_imp.values, color=g_colors[1])
g_axes[1].set_title("Random Forest: feature importance")
g_axes[1].set_xlabel("Importance (adds up to 1)")
plt.suptitle("Which features matter most?", fontsize=14)
plt.tight_layout()
plt.savefig("graph4_feature_importance.png", dpi=150)
plt.show()
 
# ---------- GRAPH 5: 5-Fold Cross-Validation score of each fold ----------
g_cv = {n: cross_val_score(models[n], X_tmp, y_tmp,
                           cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
        for n in g_names}
g_x = np.arange(5)
plt.figure(figsize=(9, 5))
plt.bar(g_x - 0.2, g_cv[g_names[0]], width=0.4, label=g_names[0], color=g_colors[0])
plt.bar(g_x + 0.2, g_cv[g_names[1]], width=0.4, label=g_names[1], color=g_colors[1])
plt.xticks(g_x, [f"Fold {i + 1}" for i in g_x])
plt.ylabel("R2 score")
plt.ylim(0, 1)
plt.title("5-Fold Cross-Validation: R2 in each fold")
plt.legend()
plt.tight_layout()
plt.savefig("graph5_cross_validation.png", dpi=150)
plt.show()
 



#Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
cm_low, cm_high = np.percentile(y_train, [33.3, 66.7])
cm_labels = ["Low", "Medium", "High"]
print(f"\nCategory cut-offs: Low < {cm_low:.1f} <= Medium < {cm_high:.1f} <= High")
 
def cm_to_group(scores_array):
    # 0 = Low, 1 = Medium, 2 = High
    return np.digitize(np.asarray(scores_array), [cm_low, cm_high])
 
cm_actual = cm_to_group(y_test)                        # real group of each test student
 
cm_fig, cm_axes = plt.subplots(1, 2, figsize=(13, 5.5))
for cm_ax, n in zip(cm_axes, models.keys()):
    cm_predicted = cm_to_group(models[n].predict(X_test))   # predicted group
    cm_matrix = confusion_matrix(cm_actual, cm_predicted, labels=[0, 1, 2])
    ConfusionMatrixDisplay(cm_matrix, display_labels=cm_labels).plot(
        ax=cm_ax, cmap="Blues", colorbar=False, values_format="d")
    cm_ax.set_title(f"{n}\nAccuracy = {accuracy_score(cm_actual, cm_predicted):.1%}")
    cm_ax.set_xlabel("Predicted group")
    cm_ax.set_ylabel("Actual group")
    print(f"\n{n}: confusion matrix (rows = actual, columns = predicted)")
    print(pd.DataFrame(cm_matrix, index=cm_labels, columns=cm_labels))
    print(classification_report(cm_actual, cm_predicted, target_names=cm_labels, digits=3))
plt.suptitle("Confusion matrices on the test set (scores grouped into Low / Medium / High)", fontsize=13)
plt.tight_layout()
plt.savefig("graph7_confusion_matrix.png", dpi=150)
plt.show()




