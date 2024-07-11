import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data=pd.read_csv("ec.csv")
df=pd.DataFrame(data)
print(df.info())
print(df.describe())
x=df.corr(numeric_only=True)
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=df,color='g')
s=sns.pairplot(df)
s.map_upper(plt.scatter)
s.map_lower(sns.violinplot)
plt.show()

print(df.columns)
df.dropna(inplace=True)
y=df[['Yearly Amount Spent']]
X=df[['Avg. Session Length','Time on App','Time on Website', 'Length of Membership']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
lm=LinearRegression()
lm.fit(X_train,y_train)
print(lm.coef_)

predictions=lm.predict(X_test)
print(predictions)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
plt.scatter(y_test, predictions)
plt.xlabel('Actual Avg. Session Length')
plt.ylabel('Predicted Avg. Session Length')
plt.title('Actual vs Predicted Avg. Session Length')
plt.show()

print(X)
print(metrics.explained_variance_score(y_test,predictions))
sns.displot((y_test-predictions),bins=50)
plt.show()
c=pd.DataFrame(lm.coef_)
d=pd.DataFrame(X.columns)
f=[d,c.transpose()]
print(pd.concat(f,axis=1))
