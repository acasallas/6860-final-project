import numpy as np
import matplotlib.pyplot as plt
from sys import argv

iteration = []
loss = []
accuracy = []

with open(argv[1]) as the_file:
    content = the_file.readlines()
    for line in content:
    	if 'iteration' in line:
    		iteration.append(float(line[11:]))
    	if 'accuracy' in line:
    		accuracy.append(float(line[11:]))
    	if 'loss' in line:
    		loss.append(float(line[6:]))

iteration = np.array(iteration)
loss = np.array(loss)
accuracy = np.array(accuracy)


print("max accuracy: " + str(np.max(accuracy)))
print("argmax: " + str(np.argmax(accuracy)))
plt.title("Results")
plt.xlabel("Iteration")
plt.ylabel("Accuracy/Loss")
plt.plot(iteration,loss,label="loss")
plt.plot(iteration,accuracy,label="accuracy")
plt.legend()
plt.show()

'''iteration2 = []
loss2 = []
accuracy2 = []

with open(argv[2]) as the_file:
    content = the_file.readlines()
    for line in content:
    	if 'iteration' in line:
    		iteration2.append(float(line[11:]))
    	if 'accuracy' in line:
    		accuracy2.append(float(line[11:]))
    	if 'loss' in line:
    		loss2.append(float(line[6:]))

iteration2 = np.array(iteration2)
loss2 = np.array(loss2)
accuracy2 = np.array(accuracy2)

print("max accuracy: " + str(np.max(accuracy)))
print("argmax: " + str(np.argmax(accuracy)))
plt.title("DropAlan vs Dropout, Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.plot(iteration2,accuracy2,label="DropAlan")
plt.plot(iteration,accuracy,label="Dropout")
plt.legend()
plt.show()'''


