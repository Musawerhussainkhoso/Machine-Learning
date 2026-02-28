from sklearn.metrics import mean_absolute_error , mean_squared_error
import numpy as np 
real_score = [90 , 60 , 80 , 100]
pred_score = [85,70,70,95]

mae = mean_absolute_error(real_score , pred_score)
mse = mean_squared_error(real_score , pred_score)
rmse = np.sqrt(mse)
print("MEE : ", mae)
print("MSE :",mse)
print("RMSE :",rmse)