import numpy as np
import matplotlib.pyplot as plt
import random
import time

#Function to generate linearly seperable points
def generate_linearly_separable_points(num_points=100, margin=0.5):
    # Generate random x-coordinates
    x_coords = np.random.uniform(-1, 1, num_points)

    # Create a linear function for separation
    # slope = (random.random()*2) -1
    slope = np.random.uniform(-10, 10)  # Generate a random slope between -10 and 10

    #No intercept for now
    intercept = 0.0
    
    # Generate y-coordinates with some margin to ensure separability
    y_coords = slope * x_coords + intercept
    y_coords += np.random.uniform(margin, 2 * margin, size=num_points) * np.sign(np.random.choice([-1, 1], size=num_points))
    
    # Classify points based on which side of the line they fall
    labels = ((y_coords > slope * x_coords + intercept).astype(int)*2) -1
    
    # Combine x and y coordinates into a single array
    points = np.column_stack((x_coords, y_coords))
    
    return np.array(points), np.array(labels),np.array(slope)

# Generate points
num_points = 100
points, labels,slope = generate_linearly_separable_points(num_points,margin=1)
plt.ion()

# Plot the points
print("Actual slope: ",slope)
fig = plt.figure(figsize=(8, 6))
for label, color in zip([-1, 1], ['blue', 'red']):
    class_points = points[labels == label]
    plt.scatter(class_points[:, 0], class_points[:, 1], label=f'Class {label}', color=color, alpha=0.7)

# Plot the separating line
x_vals = np.linspace(-1, 1, 100)
line_real = plt.plot(x_vals, slope * x_vals, color='black', linestyle='--', label='Separating Line')

#Starting Theta
theta = np.array([0.0,0.0])
 
#Plot Init
plt.legend()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Linearly Separable Points')
plt.grid(True)
plt.show()


#Function to get train error
def get_train_error(theta,points,labels):
    total = 0
    for point,label in zip(points,labels):
        if (label * np.dot(point,theta) <= 0):
            total += 1
        
    total /= points.shape[0]
    return total


#Create Line on plot for Learned Line
line_learned, = plt.plot(x_vals, (slope) * x_vals, color='green', linestyle='--', label='Learned Line')

#Itter Counter
i=0

#Loop until the solution is found
while True:
    print("Itteration: ",i)
    i+=1

    #Check if there are any errors remaining and quit if no
    if (get_train_error(theta,points,labels) == 0):
        break

    #Loop through all points and labels
    for point,label in zip(points,labels):
        #if the point is incorrectly clasified
        if (label * np.dot(point,theta) <= 0):

            #Update theta
            theta += point * label * 0.01

            #Find the slope based on normal vector theta
            slope = -(theta[0]/theta[1])

            #Update line data
            line_learned.set_ydata((slope) * x_vals)

            #Draw the figure
            fig.canvas.draw()
            fig.canvas.flush_events()

plt.ioff()
plt.show()