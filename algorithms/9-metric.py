from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, precision_recall_curve
import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\Dmitriy\PycharmProjects\classification.csv')

TP = df[(df["pred"] == 1) & (df["true"] == 1)]
FP = df[(df["pred"] == 1) & (df["true"] == 0)]
FN = df[(df["pred"] == 0) & (df["true"] == 1)]
TN = df[(df["pred"] == 0) & (df["true"] == 0)]

print(f"{len(TP)} {len(FP)} {len(FN)} {len(TN)}")

AC = accuracy_score(df["true"], df["pred"])

pr = precision_score(df["true"], df["pred"])
rec = recall_score(df["true"], df["pred"])
f1 = f1_score(df["true"], df["pred"])
print(f"{AC:.2f} {pr:.2f} {rec:.2f} {f1:.2f}")

df2 = pd.read_csv("scores.csv")
df2.head()

clf_names = df2.columns[1:]
scores = pd.Series([roc_auc_score(df2["true"], df2[clf]) for clf in clf_names], index=clf_names)

pr_scores = []
for clf in clf_names:
    pr_curve = precision_recall_curve(df2["true"], df2[clf])
    pr_scores.append(pr_curve[0][pr_curve[1] >= 0.7].max())

print(clf_names[np.argmax(pr_scores)])