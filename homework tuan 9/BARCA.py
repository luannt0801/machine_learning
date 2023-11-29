import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

filePath = "C:/Users/DELL/OneDrive - Hanoi University of Science and Technology/Desktop/Homework1/CIS419--Decision-Tree-Learning-Linear-Regression/data/multivariateData.dat"
file = open(filePath,'r')
data = np.loadtxt(file, delimiter=',')


# Normalize dữ liệu để giúp quá trình học tốt hơn
data = (data - data.mean(axis=0)) / data.std(axis=0)

# Xác định số lượng mẫu và số lượng đặc trưng
m = len(data)
n = data.shape[1] - 1  # Trừ đi cột giá nhà

# Thêm cột bias cho ma trận X
X = np.hstack((np.ones((m, 1)), data[:, :-1]))

# Khởi tạo ma trận theta
theta = np.zeros((n + 1, 1))

# Hàm tính giá trị dự đoán
def predict(X, theta):
    return X.dot(theta)

# Hàm tính chi phí (cost function)
def computeCost(X, y, theta):
    m = len(y)
    J = 1 / (2 * m) * np.sum((predict(X, theta) - y) ** 2)
    return J

# Hàm thực hiện gradient descent
def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        theta = theta - (alpha / m) * X.T.dot(predict(X, theta) - y)
        J_history[i] = computeCost(X, y, theta)

    return theta, J_history

# Lấy dữ liệu giá nhà
y = data[:, -1].reshape(-1, 1)

# Thiết lập các tham số của gradient descent
alpha = 0.01
num_iters = 1500

# Thực hiện gradient descent
theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)

# In giá trị theta tối ưu
print("Theta tối ưu:", theta.ravel())

# Vẽ đồ thị đường regressed line
plt.plot(range(1, num_iters + 1), J_history, color='b')
plt.xlabel('Số lần lặp')
plt.ylabel('Cost J')
plt.title('Học bằng gradient descent')
plt.show()

# Vẽ đồ thị surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 1], X[:, 2], y, color='r', marker='x')
ax.set_xlabel('Diện tích')
ax.set_ylabel('Số phòng')
ax.set_zlabel('Giá nhà')

x1_range = np.linspace(min(X[:, 1]), max(X[:, 1]), 10)
x2_range = np.linspace(min(X[:, 2]), max(X[:, 2]), 10)
x1_vals, x2_vals = np.meshgrid(x1_range, x2_range)
hypothesis = theta[0] + theta[1] * x1_vals + theta[2] * x2_vals
ax.plot_surface(x1_vals, x2_vals, hypothesis, color='c', alpha=0.5)
plt.show()

# Vẽ đồ thị contour plot
plt.scatter(X[:, 1], X[:, 2], c=y.ravel(), cmap='viridis', marker='x')
plt.contour(x1_vals, x2_vals, hypothesis, levels=np.arange(-1, 2, 0.5), colors='c')
plt.xlabel('Diện tích')
plt.ylabel('Số phòng')
plt.title('Contour plot của linear regression')
plt.show()
