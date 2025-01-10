import numpy as np
import time

# Function for block matrix multiplication
def block_matrix_multiply(A, B, block_size):
    """
    Multiplies two matrices A and B using block matrix multiplication.
    """
    n = A.shape[0]
    C = np.zeros((n, n))
    
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            for k in range(0, n, block_size):
                # Block multiplication
                C[i:i+block_size, j:j+block_size] += np.dot(
                    A[i:i+block_size, k:k+block_size],
                    B[k:k+block_size, j:j+block_size]
                )
    return C

# Function to measure execution time
def measure_execution_time(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    return result, end_time - start_time

# Generate random matrices
matrix_size = 512  # Size of the matrix (matrix_size x matrix_size)
block_size = 64    # Block size for block multiplication
A = np.random.rand(matrix_size, matrix_size)
B = np.random.rand(matrix_size, matrix_size)

# Measure performance of block matrix multiplication
C_block, time_block = measure_execution_time(block_matrix_multiply, A, B, block_size)

# Measure performance of regular matrix multiplication
C_regular, time_regular = measure_execution_time(np.dot, A, B)

print(time_block, time_regular)
