import numpy as np

######################################################
# Question 1
variable1 = np.array([[2, 3], [4, 5], [1, 6]])
variable2 = np.array([[2, 0], [7, 1]])

# multiplying 3x2 matrix and 2x2 matrix
result = np.dot(variable1, variable2)

print(result)

######################################################
# Questiıon 2
A = np.array([[1, 2], [4, 2]])

b = np.array([2, 7])

solution = np.linalg.solve(A, b)

# Display the solution
print("The solution is:", solution)

######################################################
# Qestion 3
x = np.array([7, 1, 3])

norm_l1 = np.linalg.norm(x, ord=1)

norm_max = np.linalg.norm(x, ord=np.inf)

norm_euclidean = np.linalg.norm(x)

norm_squared_euclidean = np.linalg.norm(x) ** 2

print("L1 Norm:", norm_l1)
print("L∞ Norm:", norm_max)
print("Euclidean Norm:", norm_euclidean)
print("Squared Euclidean Norm:", norm_squared_euclidean)

######################################################
# Qeustion 4
A = np.array([[1, 7], [2, 1]])

eigenvalues, eigenvectors = np.linalg.eig(A)

for i in range(len(eigenvalues)):
    print(f"Eigenvalue {i+1}: {eigenvalues[i]}")
    print(f"Eigenvector {i+1}: {eigenvectors[:, i]}")
    print()

######################################################
# Qestion 5
A = np.array([[1, 7], [2, 1]])

eigenvalues, eigenvectors = np.linalg.eig(A)

V = eigenvectors

Lambda = np.diag(eigenvalues)

A_reconstructed = V @ Lambda @ np.linalg.inv(V)

print("Original Matrix A:")
print(A)
print("\nReconstructed Matrix A:")
print(A_reconstructed)

######################################################
# Qestion 6
points = np.array([[1, 1], [2, 4], [3, 2], [4, 6], [5, 3]])

A = np.column_stack((np.ones(len(points)), points[:, 0]))
y = points[:, 1]

x = np.linalg.pinv(A) @ y

b, m = x[0], x[1]

print(f"The best-fit line is: y = {m:.2f}x + {b:.2f}")

######################################################
# Qestion 7
A = np.array([[1, 7], [2, 1], [4, 3]])

U, S, VT = np.linalg.svd(A)

print("Left-singular values:")
print(S)

print("\nRight-singular vectors:")
print(VT.T)

######################################################
# Qestion 8
A = np.array([[1, 7], [7, 2]])

eigenvalues, eigenvectors = np.linalg.eig(A)

Q = eigenvectors

Lambda = np.diag(eigenvalues)

A_reconstructed = Q @ Lambda @ Q.T

print("Original Matrix A:")
print(A)
print("\nReconstructed Matrix A:")
print(A_reconstructed)
