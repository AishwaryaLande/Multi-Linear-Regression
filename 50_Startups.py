import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

startup = pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/Multi Linear Regression/50_Startups.csv")
startup.head(15)
startup.columns
startup.shape
startup.head()
#startup.describe()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
startup['State']=le.fit_transform(startup['State'])
startup
startup['State'].unique()

#startup.drop(["State"],axis=1, inplace=True)
startup.columns
#startup.head()
startup.describe()
#null_columns=startup.columns[startup.isnull().any()]
#startup[null_columns].isnull().sum()
#startup.isnull()
startup.isnull().sum()

x = startup.drop(["Profit"],axis=1)
y = startup["Profit"]
y=y.astype('int')
x=x.astype('int')
plt.hist(y)
plt.hist(x)
startup.Profit.value_counts()

plt.hist(startup['spend_R_D']);plt.xlabel('spend_R_D');plt.ylabel('y');plt.title('histogram of Spend_rd')
plt.hist(startup['Administration']);plt.xlabel('Administration');plt.ylabel('y');plt.title('histogram of Administration')
plt.hist(startup['Marketing_Spend']);plt.xlabel('Marketing_Spend');plt.ylabel('y');plt.title('histogram of Marketing_Spend')
plt.hist(startup['State']);plt.xlabel('State');plt.ylabel('y');plt.title('histogram of State')

sns.boxplot(startup.spend_R_D)
sns.boxplot(startup.Administration)
sns.boxplot(startup.Marketing_Spend)
sns.boxplot(startup.Profit)

sns.pairplot((startup),hue='Profit')

# Normal Q- plot
plt.plot(startup);plt.legend(list(startup.columns))
spend_R_D= np.array(startup['spend_R_D'])
Administration = np.array(startup['Administration'])
Marketing_Spend = np.array(startup['Marketing_Spend'])
State = np.array(startup['State'])
Profit = np.array(startup['Profit'])

from scipy import stats
stats.probplot(spend_R_D, dist='norm', plot=plt);plt.title('Probability Plot of spend_R_D')
stats.probplot(Administration, dist='norm', plot=plt);plt.title('Probability Plot of Administration')
stats.probplot(Marketing_Spend, dist='norm', plot=plt);plt.title('Probability Plot of Marketing_Spend')
stats.probplot(State, dist='norm', plot=plt);plt.title('Probability Plot of State')
stats.probplot(y, dist='norm', plot=plt);plt.title('Probability Plot of Profit')

startup.corr()

sns.heatmap(startup.corr(),annot=True) 

import statsmodels.formula.api as smf # for regression model
         
# Preparing model                  
model1 = smf.ols('Profit~spend_R_D+Marketing_Spend+State',data=startup).fit() # regression model              
model1.params
model1.summary() ## 0.950

import statsmodels.api as sm
sm.graphics.influence_plot(model1)

# index 48 AND 49 is showing high influence so we can exclude that entire row
strt_new=startup.drop(startup.index[[45,49]],axis=0)

# Preparing model                  
model_new = smf.ols('Profit~spend_R_D+Marketing_Spend+State',data = strt_new).fit()    

# Getting coefficients of variables        
model_new.params

# Summary
model_new.summary()  ## 0.964

print(model_new.conf_int(0.01)) # 99% confidence level

# Predicted values of MPG 
profit_pred = model_new.predict(strt_new[['spend_R_D','Marketing_Spend','State','Profit']])
profit_pred

strt_new.head()
# calculating VIF's values of independent variables
rsq_rd = smf.ols('spend_R_D~Marketing_Spend+State+Profit',data=strt_new).fit().rsquared  
vif_rd = 1/(1-rsq_rd) 
vif_rd   #25.294416565709174

rsq_Profit = smf.ols('Profit~spend_R_D+Marketing_Spend+State',data=strt_new).fit().rsquared  
vif_Profit = 1/(1-rsq_Profit)
vif_Profit    #27.862205055268195

rsq_marketspend = smf.ols('Marketing_Spend~spend_R_D+Profit+State',data=strt_new).fit().rsquared  
vif_marketspend = 1/(1-rsq_marketspend) 
vif_marketspend  #2.234

rsq_state = smf.ols('State~Marketing_Spend+spend_R_D+Profit+State',data=strt_new).fit().rsquared  
vif_state = 1/(1-rsq_state) 
vif_state ## 2.184911926504765

# Storing vif values in a data frame
d1 = {'Variables':['spend_R_D','Profit','Marketing_Spend','State'],'VIF':[vif_rd,vif_Profit,vif_marketspend,vif_state]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As Profit is having higher VIF value, we are not going to include this prediction model
# Added varible plot 
sm.graphics.plot_partregress_grid(model_new)

# added varible plot for spend_rd is not showing any significance 

# final model
final_model= smf.ols('spend_R_D~Marketing_Spend+Profit+State',data = strt_new).fit()
final_model.params
final_model.summary() 
# As we can see that r-squared value is 0.96
profit_pred = model_new.predict(strt_new)
profit_pred

import statsmodels.api as sm
# added variable plot for the final model
sm.graphics.plot_partregress_grid(model_new)

#  Linearity
# Observed values VS Fitted values
plt.scatter(strt_new.Profit,profit_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

# Residuals VS Fitted Values 
plt.scatter(profit_pred,model_new.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

#Normality plot for residuals
# histogram
plt.hist(model_new.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(model_new.resid_pearson, dist="norm", plot=pylab)

#Homoscedasticity 
# Residuals VS Fitted Values 
plt.scatter(profit_pred,model_new.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

#Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
strt_train,strt_test  = train_test_split(strt_new,test_size = 0.2) 
# preparing the model on train data 
model_train = smf.ols("Profit~spend_R_D+Marketing_Spend+State",data=strt_train).fit()
# train_data prediction
train_pred = model_train.predict(strt_train)
# train residual values 
train_resid  = train_pred - strt_train.Profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
train_rmse # 6696.450155285494
# prediction on test data set 
test_pred = model_train.predict(strt_test)
# test residual values test_resid  = test_pred - strt_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
test_rmse #9125.553069162244
