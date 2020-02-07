# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# loading the data
comp = pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/Multi Linear Regression/Computer_Data (2).csv")

comp.drop('Unnamed: 0',axis=1,inplace=True)

comp.head(40) 
comp.drop_duplicates(keep='first',inplace=True)
comp.isnull().sum()
comp.columns

comp = pd.get_dummies(comp)
comp.columns
comp.drop(["cd_no","multi_no","premium_no"],axis=1,inplace=True)

plt.hist(comp.hd)
plt.hist(np.log(comp.hd)) # to normalize

plt.hist(comp.price)
plt.hist(np.log(comp.price))

plt.hist(comp.speed)
plt.hist(np.log(comp.speed))

plt.hist(comp.ram)
plt.hist(np.log(comp.ram))

plt.hist(comp.screen)

plt.hist(comp.cd_yes)
plt.hist(comp.multi_yes)
plt.hist(comp.premium_yes)
 
sns.boxplot(comp.hd)      #outlier
sns.boxplot(comp.price)   #outlier
sns.boxplot(comp.speed)   #outlier
sns.boxplot(comp.ram)     #outlier
sns.boxplot(comp.screen)  #outlier

sns.pairplot(comp)
# Correlation matrix 
comp.corr()
corr=comp.corr()
corr
sns.heatmap(corr,annot=True)
comp.drop('ads',inplace=True,axis=1)
corr=comp.corr()
sns.heatmap(corr,annot=True)
 # columns names
comp.columns

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model         
# Preparing model                  
model1 = smf.ols('hd~ram+price+speed+screen+trend',data=comp).fit() # regression model
# Getting coefficients of variables               
model1.params
# Summary
model1.summary()    #R-squared: 0.766
# preparing model based only on hd
model_hd=smf.ols('hd~ram',data = comp).fit()  
model_hd.summary() #  0.605
# p-value <0.05 .. It is significant 
# Preparing model based only on ram
model_ram=smf.ols('hd~price',data = comp).fit()  
model_ram.summary() # 0.185

model_scr=smf.ols('hd~screen',data = comp).fit()  
model_scr.summary() #  0.054

model_trd=smf.ols('hd~trend',data = comp).fit()  
model_trd.summary() #  0.334

model_sp=smf.ols('hd~speed',data = comp).fit()  
model_sp.summary() #0.139

# Preparing model based 
model_new=smf.ols('hd~ram+trend+price',data = comp).fit()  
model_new.summary() # 0.766
 
# influence index plots

import statsmodels.api as sm
sm.graphics.influence_plot(model1)
# index 3783 is showing high influence so we can exclude that entire row
# Studentized Residuals = Residual/standard deviation of residuals
comp_new=comp.drop(comp.index[[3783]],axis=0)
# Preparing model                  
model1_new = smf.ols('hd~ram+price+trend',data=comp).fit() # regression model
# Getting coefficients of variables        
model1_new.params
# Summary
model1_new.summary() 
# Confidence values 99%
print(model1_new.conf_int(0.01)) # 99% confidence level
# Predicted values of MPG 
mpg_pred = model1_new.predict(comp_new[['hd','ram','price','trend']])
mpg_pred

comp_new.head()
# calculating VIF's values of independent variables
rsq_hd = smf.ols('hd~ram+price+trend',data=comp_new).fit().rsquared  
vif_hd = 1/(1-rsq_hp) # 4.313949957911793

rsq_ram = smf.ols('ram~hd+price+trend',data=comp_new).fit().rsquared  
vif_ram = 1/(1-rsq_ram) # 3.4409659223445144

rsq_pr = smf.ols('price~hd+ram+trend',data=comp_new).fit().rsquared  
vif_pr = 1/(1-rsq_pr) #  2.331559570308275

rsq_tr = smf.ols('trend~hd+ram+price',data=comp_new).fit().rsquared  
vif_tr = 1/(1-rsq_tr) #   2.402347294286398

# Storing vif values in a data frame
d1 = {'Variables':['hd','ram','price','trend'],'VIF':[vif_hd,vif_ram,vif_pr,vif_tr]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# Added varible plot 
sm.graphics.plot_partregress_grid(model1_new)

# added varible plot for weight is not showing any significance 
# final model
final_ml= smf.ols('hd~ram+price+trend',data = comp_new).fit()
final_ml.params
final_ml.summary() #  0.768

mpg_pred = final_ml.predict(comp_new)

import statsmodels.api as sm
# added variable plot for the final model
sm.graphics.plot_partregress_grid(final_ml)


#Linearity
# Observed values VS Fitted values
plt.scatter(comp_new.hd,mpg_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

# Residuals VS Fitted Values 
plt.scatter(mpg_pred,final_ml.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

#Normality plot for residuals 
# histogram
plt.hist(final_ml.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 
import pylab          
import scipy.stats as st
# Checking Residuals are normally distributed
st.probplot(final_ml.resid_pearson, dist="norm", plot=pylab)
#Homoscedasticity 
# Residuals VS Fitted Values 
plt.scatter(mpg_pred,final_ml.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

#Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
comp_train,comp_test  = train_test_split(comp_new,test_size = 0.2) # 20% size

# preparing the model on train data 

model_train = smf.ols("hd~ram+price+trend",data=comp_train).fit()

# train_data prediction
train_pred = model_train.predict(comp_train)

# train residual values 
train_resid  = train_pred - comp_train.hd

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))   #121.35067230704148
train_rmse
# prediction on test data set 
test_pred = model_train.predict(comp_test)

# test residual values 
test_resid  = test_pred - comp_test.hd

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid)) #134.45609495765612
test_rmse
