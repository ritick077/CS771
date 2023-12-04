import numpy as np
import matplotlib.pyplot as plt

class LandmarkRegression:
    def __init__(self, data_location):
        self.training_data_x, self.training_data_y, self.testing_data_x, self.testing_data_y = self.load_data(data_location)
        self.num_landmarks_values = [2, 5, 20, 50, 100]

    def load_data(self, location):
        testing_data = np.genfromtxt(location + '/ridgetest.txt', delimiter='  ')
        training_data = np.genfromtxt(location + '/ridgetrain.txt', delimiter='  ')
        return training_data[:, 0], training_data[:, 1], testing_data[:, 0], testing_data[:, 1]

    def landmark_kernel(self, x, y):
        return np.exp(-0.1 * np.square(x.reshape((-1, 1)) - y.reshape((1, -1))))

    def train_predict_plot(self, num_landmarks):
        landmark_samples = np.random.choice(self.training_data_x, num_landmarks, replace=False)
        identity_matrix = np.eye(num_landmarks)
        kernel_train = self.landmark_kernel(self.training_data_x, landmark_samples)

        weights = np.dot(np.linalg.inv(np.dot(kernel_train.T, kernel_train) + 0.1 * identity_matrix),
                        np.dot(kernel_train.T, self.training_data_y.reshape((-1, 1))))

        kernel_test = self.landmark_kernel(self.testing_data_x, landmark_samples)
        y_pred = np.dot(kernel_test, weights)

        rmse = np.sqrt(np.mean(np.square(self.testing_data_y.reshape((-1, 1)) - y_pred)))
        print('RMSE for num_landmarks = ' + str(num_landmarks) + ' is ' + str(rmse))

        plt.figure(num_landmarks)
        plt.title('Num Landmarks = ' + str(num_landmarks) + ', rmse = ' + str(rmse))
        plt.plot(self.testing_data_x, y_pred, '*',color='orange')
        plt.plot(self.testing_data_x, self.testing_data_y, '*',color='#6495ED' )

    def run(self):
        for num_landmarks in self.num_landmarks_values:
            self.train_predict_plot(num_landmarks)

        plt.show()

landmark_model = LandmarkRegression('data')
landmark_model.run()
