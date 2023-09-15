##Import necessary packages and modules:
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import cvxpy as cp

##Volatility
#Import variables for HAR Volatility Model
HAR_Vol = pd.read_excel("Data/HAR_Vol_Variables.xlsx")
#Construct HAR Volatility Model
y_1 = HAR_Vol['RV']
independent_variables_1 = HAR_Vol[['RV_d','RV_w','RV_m']]
X_1 = sm.add_constant(independent_variables_1)
vol_model = sm.OLS(y_1,X_1).fit()
#print(vol_model.summary())

##Correlation
#Import variables for HAR Correlation Model
HAR_Corr = pd.read_excel("Data/HAR_Corr_Variables.xlsx")
#Construct HAR Correlation Model
y_2 = HAR_Corr['Corr'] - HAR_Corr['Corr_bar']
independent_variables_2 = pd.DataFrame()
independent_variables_2['Corr_w - Corr_bar'] = HAR_Corr['Corr_w'] - HAR_Corr['Corr_bar']
independent_variables_2['Corr_m - Corr_bar'] = HAR_Corr['Corr_m'] - HAR_Corr['Corr_bar']
corr_model = sm.OLS(y_2,independent_variables_2).fit()
#print(corr_model.summary())

##Forecasting and errors
#Import test datasets
OneDayVol = pd.read_excel("Data/1DayVol_Test.xlsx", index_col=0)
WeekVol = pd.read_excel("Data/5DayVol_Test.xlsx", index_col=0)
MonthVol = pd.read_excel("Data/21DayVol_Test.xlsx", index_col=0)
ActualVol = pd.read_excel("Data/RealizedVolatility_Test.xlsx", index_col=0)

Corr_Bar = pd.read_excel("Data/Correlations_Overall_Test.xlsx", index_col=0)
WeekCorr = pd.read_excel("Data/Correlations_5Day_Test.xlsx", index_col=0)
MonthCorr = pd.read_excel("Data/Correlations_21Day_Test.xlsx", index_col=0)
ActualCorr = pd.read_excel("Data/Correlations_Test.xlsx", index_col=0)

#Generate Volatility forecasts
VolForecast = vol_model.params[0] + vol_model.params[1]*OneDayVol + vol_model.params[2]*WeekVol + vol_model.params[3]*MonthVol
#Calculate forecast errors (L2 norm)
VolForecastingError = np.sqrt(((ActualVol - VolForecast)**2).sum(axis = 1))

#Generate correlation forecasts
CorrForecast = (1 - corr_model.params[0] - corr_model.params[1])*Corr_Bar + corr_model.params[0]*WeekCorr + corr_model.params[1]*MonthCorr
#Calculate correlation errors (L2 norm)
CorrForecastingError = np.sqrt(((ActualCorr - CorrForecast)**2).sum(axis = 1))

#Graph Vol Losses
plt.plot(VolForecastingError.iloc[:530]) #pre oil-shock for better viewability
plt.xlabel('Date')
plt.ylabel('L2 Distance')
plt.title('Volatility Errors')
#plt.show()

plt.plot(VolForecastingError.iloc[530:600]) #oil shock
plt.xlabel('Date')
plt.ylabel('L2 Distance')
plt.title('Volatility Errors')
#plt.show()

plt.plot(VolForecastingError.iloc[600:]) #post oil shock
plt.xlabel('Date')
plt.ylabel('L2 Distance')
plt.title('Volatility Errors')
#plt.show()

#Graph Correlation Losses
plt.plot(CorrForecastingError)
plt.xlabel('Date')
plt.ylabel('L2 Distance')
plt.title('Correlation Errors')
#plt.show()

##Minimum Variance Portfolio Loss Function
#import returns 2018-2022
returns = pd.read_excel("Data/DailyReturns_Test.xlsx",index_col=0)
#Create Covariance Matrices

#Volatility Diagonals: construct D matrix
Diagonals = []
for i in range(0,len(VolForecast)):
    matrix = np.zeros((4,4))
    for j in range(0,4):
        matrix[j,j] = VolForecast.iloc[i,j]
    Diagonals.append(matrix)

#Populate Correlation matrices: construct R matrix
Correlations = []
for i in range(0,len(CorrForecast)):
    matrix = np.zeros((4,4))
    matrix[1,0] = CorrForecast.iloc[i,0]
    matrix[2,0] = CorrForecast.iloc[i,1]
    matrix[3,0] = CorrForecast.iloc[i,2]
    matrix[2,1] = CorrForecast.iloc[i,3]
    matrix[3,1] = CorrForecast.iloc[i,4]
    matrix[3,2] = CorrForecast.iloc[i,5]
    matrix[0,1] = matrix[1,0]
    matrix[0,2] = matrix[2,0]
    matrix[0,3] = matrix[3,0]
    matrix[1,2] = matrix[2,1]
    matrix[1,3] = matrix[3,1]
    matrix[2,3] = matrix[3,2]
    for j in range(0,4):
        matrix[j,j] = 1
    Correlations.append(matrix)

#DRD = Covariance matrix: matrix multiply DRD
Covariances = []
for i in range(0,len(Correlations)):
   Covariances.append(Diagonals[i] @ Correlations[i] @ Diagonals[i])

#Optimize for weights: objective is to construct minimum variance portfolio

weights = cp.Variable(4)
weightslist = []
for i in range(0,len(CorrForecast)):
    variance = cp.quad_form(weights, Covariances[i])
    constraints = [cp.sum(weights) == 1, weights >= 0]
    objective = cp.Minimize(variance)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    optimal_weights = weights.value
    weightslist.append(optimal_weights)

#convert weights list to df
weightsdataframe = pd.DataFrame(weightslist,index = CorrForecast.index, columns = VolForecast.columns)
#calculate daily portfolio returns
portfolioreturns = (weightsdataframe*returns.iloc[:-4]).sum(axis=1)
#calculate cumulative daily portfolio returns
cumulativeportfolioreturns = np.exp((np.cumsum(portfolioreturns)))-1
#in terms of $100 starting investment in 2018
portfoliovalue = 100* (1+cumulativeportfolioreturns)


##Baseline portfolio - historical data: for comparision against HAR-DRD
#Volatility Diagonals: construct D matrix
Diagonals_b = []
for i in range(0,len(OneDayVol)):
    matrix = np.zeros((4,4))
    for j in range(0,4):
        matrix[j,j] = OneDayVol.iloc[i,j]
    Diagonals_b.append(matrix)

#Populate Correlation matrices: construct R matrix
Correlations_b = []
for i in range(0,len(WeekCorr)):
    matrix = np.zeros((4,4))
    matrix[1,0] = WeekCorr.iloc[i,0]
    matrix[2,0] = WeekCorr.iloc[i,1]
    matrix[3,0] = WeekCorr.iloc[i,2]
    matrix[2,1] = WeekCorr.iloc[i,3]
    matrix[3,1] = WeekCorr.iloc[i,4]
    matrix[3,2] = WeekCorr.iloc[i,5]
    matrix[0,1] = matrix[1,0]
    matrix[0,2] = matrix[2,0]
    matrix[0,3] = matrix[3,0]
    matrix[1,2] = matrix[2,1]
    matrix[1,3] = matrix[3,1]
    matrix[2,3] = matrix[3,2]
    for j in range(0,4):
        matrix[j,j] = 1
    Correlations_b.append(matrix)

#DRD = Covariance matrix: matrix multiply DRD
Covariances_b = []
for i in range(0,len(Correlations_b)):
   Covariances_b.append(Diagonals_b[i] @ Correlations_b[i] @ Diagonals_b[i])

#Optimize for weights : objective is to construct minimum variance portfolio

weights = cp.Variable(4)
weightslist_b = []
for i in range(0,len(WeekCorr)):
    variance = cp.quad_form(weights, Covariances_b[i])
    constraints = [cp.sum(weights) == 1, weights >= 0]
    objective = cp.Minimize(variance)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    optimal_weights = weights.value
    weightslist_b.append(optimal_weights)

#similar manipulations to weights and returns as we did earlier
weightsdataframe_b = pd.DataFrame(weightslist_b,index = WeekCorr.index, columns = OneDayVol.columns)
portfolioreturns_b = (weightsdataframe_b*returns.iloc[:-4]).sum(axis=1)
cumulativeportfolioreturns_b = np.exp((np.cumsum(portfolioreturns_b)))-1
portfoliovalue_b = 100* (1+cumulativeportfolioreturns_b)

#plot both portfolio growths
plt.plot(portfoliovalue, label = 'HAR-DRD')
plt.plot(portfoliovalue_b, label = 'Historical')
plt.xlabel('Date')
plt.ylabel('$')
plt.legend()
plt.title('Portfolio Value')
#plt.show()

#compute yearwise annualized volatility using both approaches
portfolioreturns_reset = portfolioreturns.reset_index()
portfolioreturns_b_reset = portfolioreturns_b.reset_index()
portfolioreturns_reset.columns = ['Date','Returns']
portfolioreturns_b_reset.columns = ['Date','Returns']

print(portfolioreturns_reset.groupby(portfolioreturns_reset.Date.dt.year)['Returns'].std()*np.sqrt(252))
print(portfolioreturns_b_reset.groupby(portfolioreturns_b_reset.Date.dt.year)['Returns'].std()*np.sqrt(252))