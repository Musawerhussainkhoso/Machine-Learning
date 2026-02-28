from sklearn.metrics import confusion_matrix

x_true = [1, 0, 1, 1, 0]
x_pred = [1, 0, 0, 1, 0]
cm = confusion_matrix(x_true , x_pred)
print(cm)