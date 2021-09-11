import csv
import numpy as np

def cost_function(theta,x,y):
    hypotheses = x.dot(theta)
    cost = (len(y)/2) * np.sum(np.square(hypotheses-y))
    return cost

def stocashtic_gradient_descent(theta,x,y,learning_rate=0.01,iters=10):
    calculated_costs = np.zeros(iters)
    
    ylen = len(y)
    for iter in range(iters):
        cost = 0.0
        for i in range(ylen):
            random_index = np.random.randint(0, ylen)
            xhat = x[random_index,0].reshape(1,x.shape[1])
            yhat = y[random_index,0].reshape(1,1)
            hypothesis = np.dot(xhat,theta)

            theta -= (1/ylen)*learning_rate*(xhat.T.dot((hypothesis-yhat)))
            cost += cost_function(theta,xhat,yhat)
        calculated_costs[iter] = cost
        
    return theta, calculated_costs

def batch_gradient_descent(theta,x,y,learning_rate=0.01,iters=100):
    calculated_costs = np.zeros(iters)
    calculated_thetas = np.zeros((iters,2))

    ylen = len(y)
    for iter in range(iters):
        hypothesis = np.dot(x,theta)
        theta -= (1/ylen)*learning_rate*(x.T.dot((hypothesis - y)))

        calculated_costs[iter] = cost_function(theta,x,y)        
        calculated_thetas[iter,0] = theta.T[0]
        calculated_thetas[iter,1] = theta.T[1]
        
    return theta, calculated_costs, calculated_thetas

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

x = []
y = []
iter = 0
with open('Video_Game_Sales_Revised.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if isfloat(row[0]) and isfloat(row[1]):
            x.append([])
            y.append([])
            x[iter].append(float(row[0]))
            y[iter].append(float(row[1]))
            iter+=1

x = np.array(x)
y = np.array(y)

lr = 0.01
num_iter = 10
theta = np.random.randn(1,2)
theta,calculated_costs = stocashtic_gradient_descent(theta,x,y,lr,num_iter)

print("Stocashtic Gradient Descent")
print("Theta 0 :",theta[0][0]," Theta1 :",theta[0][1])
print("Calculated Cost :",calculated_costs[len(calculated_costs)-1]/len(calculated_costs))

lr = 0.01
num_iter = 1000
theta = np.random.randn(1,2)
theta,calculated_costs,calculated_thetas = batch_gradient_descent(theta,x,y,lr,num_iter)

final_cost = calculated_costs[len(calculated_costs)-1]/len(calculated_costs)
print("\nBatch Gradient Descent")
print("Theta 0 :",theta[0][0]," Theta1 :",theta[0][1])
print("Calculated Cost :",final_cost)
