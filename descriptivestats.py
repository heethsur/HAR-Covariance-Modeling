import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

returnsdata = pd.read_excel("Data/DailyReturns.xlsx", index_col=0 )

print('Mean:',returnsdata.mean())
print('Std:',returnsdata.std())
print('Skew:',returnsdata.skew())
print('Kurtosis:',returnsdata.kurtosis())

aggregated = np.exp((np.cumsum(returnsdata)))-1
dollargrowth = 100* (1+aggregated) 
#print(dollargrowth)

plt.plot(dollargrowth['US GB'], label = 'US TY - 10Yr')
plt.plot(dollargrowth['US Equity'], label = 'US SP500')
plt.plot(dollargrowth['WTI'], label = 'WTI Crude')
plt.plot(dollargrowth['Gold'], label = 'Gold')
plt.title('Cumulative (Log) Returns 2002-2022')
plt.xlabel('Date')
plt.ylabel('$')
vertical_line_date = datetime.datetime(2018, 1, 1)
plt.axvline(x=vertical_line_date, color='black', linestyle='--', label='Train-Test split')
plt.legend()
plt.show()