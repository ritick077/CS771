import numpy as np
import matplotlib.pyplot as plt

class RidgeRegressionModified:
    def __init__(self, data_location):
        self.train_data_x, self.train_data_y, self.test_data_x, self.test_data_y = self.load_data(data_location)
        self.kernel_matrix = self.ridge_kernel(self.train_data_x, self.train_data_x)
        self.regularization_parameters = [0, 0.1, 1, 10, 100]
        self.identity_matrix = np.eye(self.train_data_x.shape[0])

    def load_data(self, loc):
        train_data = np.genfromtxt(loc + '/ridgetrain.txt', delimiter='  ')
        test_data = np.genfromtxt(loc + '/ridgetest.txt', delimiter='  ')
        return train_data[:, 0], train_data[:, 1], test_data[:, 0], test_data[:, 1]

    def ridge_kernel(self, x, y):
        return np.exp(-0.1 * np.square(x.reshape((-1, 1)) - y.reshape((1, -1))))

    def train_predict_plot(self, lambda_value):
        alpha = np.dot(np.linalg.inv(self.kernel_matrix + lambda_value * self.identity_matrix),
                       self.train_data_y.reshape((-1, 1)))
        kernel_test = self.ridge_kernel(self.train_data_x, self.test_data_x)
        y_pred = (np.dot(alpha.T, kernel_test)).reshape((-1, 1))

        rmse = np.sqrt(np.mean(np.square(self.test_data_y.reshape((-1, 1)) - y_pred)))
        print('RMSE value for lambda = ' + str(lambda_value) + ' is ' + str(rmse))

        plt.figure(lambda_value)
        plt.title('lambda = ' + str(lambda_value) + ', rmse = ' + str(rmse))
        plt.plot(self.test_data_x, y_pred, '*', color = 'orange')
        plt.plot(self.test_data_x, self.test_data_y, '*', color='#6495ED')

    def run(self):
        for lambda_val in self.regularization_parameters:
            self.train_predict_plot(lambda_val)

        plt.show()

ridge_model_modified = RidgeRegressionModified('data')
ridge_model_modified.run()
