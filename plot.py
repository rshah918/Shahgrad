import matplotlib.pyplot as plt
import numpy as np

# Read the output values from the file
with open("output_values.txt", "r") as f:
    output_values = [float(line.strip()) for line in f.readlines()]

# Generate x values (indices)
x_values = np.arange(0,1, 0.001)
print(len(x_values))
print(len(output_values))

# Plot the output values as dots
plt.scatter(x_values, output_values, marker='.', label='Output Values')

# Plot y = x^2 curve
plt.plot(x_values, np.square(x_values), label='$y = x^2$')

plt.xlabel('Index (I)')
plt.ylabel('Value')
plt.title('Output Values and $y = x^2$ Curve')
plt.legend()
plt.grid(True)
plt.show()
