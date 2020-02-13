import pandas as pd 
import numpy as np 
import seaborn as snp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn import metrics

class Mediclaim:
	def __init__(self, file_name):
		self.data = pd.read_csv(file_name)
		self.display_head()	
		self.train_test_assign()

	def display_head(self):
		print(self.data.head())

	def train_test_assign(self):
		self.X = self.data[['age', 'bmi', 'children']]
		self.y = self.data['charges']
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( self.X, 
																				 self.y, 
																				 test_size=0.3, 
																				 random_state=40
																				)
		self.fit_training_data()

	def fit_training_data(self):
		self.lm = LinearRegression()
		self.lm.fit(self.X_train, self.y_train)
		self.prediction()

	def prediction(self):
		self.predict_value = self.lm.predict(self.X_test)
		self.versus_graph()

	def versus_graph(self):
		snp.scatterplot(self.y_test, self.predict_value)
		plt.xlabel('True Test Values')
		plt.ylabel('Predicted Values')
		self.error_evaluation()

	def error_evaluation(self):
		print('MAE:', metrics.mean_absolute_error(self.y_test, self.predict_value))
		print('MSE:', metrics.mean_squared_error(self.y_test, self.predict_value))
		print('RMSE:', np.sqrt(metrics.mean_squared_error(self.y_test, self.predict_value)))


Mediclaim('insurance.csv')
plt.show()
