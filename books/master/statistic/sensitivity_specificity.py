# suppose we make a prediction, result as:

#         |  实际阳性  | 实际阴性
# --------------------------------
# 测试阳性 |  20 (TP)  | 180 (FP)
# --------------------------------
# 测试阴性 |  10 (FN)  | 1820 (TN)
# --------------------------------

TP = 20
FP = 180
FN = 10
TN = 1820

# 敏感度：所有真值中，有多大的比率被预测为真
# 敏感度越高，说明 漏检 越少
sensitivity = TP / (TP + TN)
# 敏感度也称为召回率
recall = sensitivity

# 特异度：所有假值中，有多大的比率预测为假
# 特异度越高，说明 误检 越少
specificity = TN / (TN + FP)

print("Recall: %.3f" % recall)
print("Specificity: %.3f" % specificity)


accuracy = (TP + TN) / (TP + TN + FN + FP)

print("Accuracy: %.3f" % accuracy)

F_score = 2 * sensitivity * specificity / (sensitivity + specificity)
print("F-score: %.3f" % F_score)