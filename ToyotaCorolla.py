# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# loading the data
toyota = pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/Multi Linear Regression/ToyotaCorolla.csv",header=0,encoding = 'unicode_escape')

toyota.drop(["Id","Model","Mfg_Month","Mfg_Year","Fuel_Type","Met_Color","Color","Automatic","Cylinders","Mfr_Guarantee","BOVAG_Guarantee","Guarantee_Period","ABS","Airbag_1","Airbag_2","Airco","Automatic_airco","Boardcomputer","CD_Player","Central_Lock","Powered_Windows","Power_Steering","Radio","Mistlamps","Sport_Model","Backseat_Divider","Metallic_Rim","Radio_cassette","Tow_Bar"],axis=1,inplace=True)
toyota.drop_duplicates(keep='first',inplace=True)
toyota.isnull().sum()
toyota.shape

toyota.head(10)
#toyota.info
toyota.columns

sns.boxplot(toyota.Price)
sns.boxplot(toyota.Age_08_04)
sns.boxplot(toyota.KM)
sns.boxplot(toyota.HP)
sns.boxplot(toyota.cc)
sns.boxplot(toyota.Doors)
sns.boxplot(toyota.Gears)
sns.boxplot(toyota.Quarterly_Tax)
sns.boxplot(toyota.Weight)

sns.pairplot((toyota),hue='Price')

corr = toyota.corr()
corr
sns.heatmap(corr,annot=True)

# Correlation matrix 
toyota.corr()

# we see there exists High collinearity between input variables especially 
                             
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
# Preparing model                  
model1 = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota).fit() # regression model
# Getting coefficients of variables               
model1.params 

# Summary
model1.summary()   #R-squared value 0.863

# preparing model based only on KM
model_k=smf.ols('Price~KM',data = toyota).fit()  
model_k.summary() #  0.324

# Preparing model based only on HP
model_h=smf.ols('Price~HP',data = toyota).fit()  
model_h.summary() # 0.099

# Preparing model based only on KM & HP
model_kh=smf.ols('Price~KM+HP',data = toyota).fit()  
model_kh.summary() # 0.342

# Checking whether data has any influential values 
# influence index plots
import statsmodels.api as sm
sm.graphics.influence_plot(model1)
# index 80 is showing high influence so we can exclude that entire row

# Studentized Residuals = Residual/standard deviation of residuals

toyota=toyota.drop(toyota.index[[80]],axis=0)

# Preparing model                  
model_new = smf.ols('Price~KM+HP+cc+Doors+Gears+Weight',data = toyota).fit()    

# Getting coefficients of variables        
model_new.params

# Summary
model_new.summary() # 0.653

# Confidence values 99%
print(model_new.conf_int(0.01)) # 99% confidence level

# Predicted values of MPG 
mpg_pred = model_new.predict(toyota[['KM','HP','cc','Doors','Gears','Weight']])
mpg_pred
toyota.head()
# calculating VIF's values of independent variables
rsq_hp = smf.ols('HP~KM+cc+Doors+Gears+Price',data=toyota).fit().rsquared  
vif_hp = 1/(1-rsq_hp) # 1.2376417349926667

rsq_km = smf.ols('KM~HP+cc+Doors+Gears+Price',data=toyota).fit().rsquared  
vif_km = 1/(1-rsq_km) # 1.2376417349926667

rsq_cc = smf.ols('cc~HP+KM+Doors+Gears+Price',data=toyota).fit().rsquared  
vif_cc = 1/(1-rsq_cc) # 1.3700326895663466

rsq_pr = smf.ols('Price~HP+KM+Doors+Gears+cc',data=toyota).fit().rsquared  
vif_pr = 1/(1-rsq_pr) # 1.880093472259839

rsq_gr = smf.ols('Gears~HP+KM+Doors+Price+cc',data=toyota).fit().rsquared  
vif_gr = 1/(1-rsq_gr) #  1.1014367994079557

rsq_dr = smf.ols('Doors~HP+KM+Gears+Price+cc',data=toyota).fit().rsquared  
vif_dr = 1/(1-rsq_dr) # 1.091523235730342

# Storing vif values in a data frame
d1 = {'Variables':['KM','HP','Doors','Price','Gears','cc'],'VIF':[vif_hp,vif_km,vif_pr,vif_cc,vif_gr,vif_dr]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# Added varible plot 
sm.graphics.plot_partregress_grid(model_new)

# final model
final_model= smf.ols('Price~KM+cc+Gears+Doors',data = toyota).fit()
final_model.params
final_model.summary() # 0.466

mpg_pred = final_model.predict(toyota)

import statsmodels.api as sm
# added variable plot for the final model
sm.graphics.plot_partregress_grid(final_model)

######  Linearity #########
# Observed values VS Fitted values
plt.scatter(toyota.Price,mpg_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

# Residuals VS Fitted Values 
plt.scatter(mpg_pred,final_model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

########    Normality plot for residuals ######
# histogram
plt.hist(final_model.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(final_model.resid_pearson, dist="norm", plot=pylab)

############ Homoscedasticity #######
# Residuals VS Fitted Values 
plt.scatter(mpg_pred,final_model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
toyota_train,toyota_test  = train_test_split(toyota,test_size = 0.2) # 20% size

# preparing the model on train data 
model_train = smf.ols("Price~KM+cc+Gears+Doors",data=toyota_train).fit()
# train_data prediction
train_pred = model_train.predict(toyota_train)
# train residual values 
train_resid  = train_pred - toyota_train.Price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid)) #2628.8365340108667
train_rmse #2628.8365340108667
# prediction on test data set 
test_pred = model_train.predict(toyota_test)
# test residual values 
test_resid  = test_pred - toyota_test.Price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
test_rmse #2652.38646537416
