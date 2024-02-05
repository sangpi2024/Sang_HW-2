import numpy as np

def GaussSeidel(Aaug, x, Niter=15):
    """
    Gauss-Seidel method to estimate the solution to a set of linear equations Ax = b.
    Aaug is an augmented matrix containing [A | b] where A is NxN, b is Nx1.
    x is the initial guess, and Niter is the number of iterations to perform.
    Returns the final estimate of x.
    """
    # Separate the matrix A and vector b from the augmented matrix Aaug
    A = Aaug[:, :-1]
    b = Aaug[:, -1]

    # Number of equations
    N = A.shape[0]

    # Perform iterations
    for _ in range(Niter):
        for i in range(N):
            # Sum over all elements of the row excluding the diagonal
            sigma = sum(A[i, j] * x[j] for j in range(N) if j != i)
            # Update x[i] considering the current iteration's other x values
            x[i] = (b[i] - sigma) / A[i, i]
    return x

def main():
    # Define the augmented matrices for both systems of equations
    Aaug1 = np.array([[3, -1, 0, 2],
                      [1, 4, 1, 12],
                      [2, 1, 2, 10]], dtype=float)

    Aaug2 = np.array([[1, -10, 2, 0, 2],
                      [3, 1, 4, 0, 12],
                      [9, 2, 3, 0, 21],
                      [-1, 2, 7, 3, 37]], dtype=float)

    # Initial guesses for x, could be zeros or any other reasonable guess
    x0_1 = np.zeros(Aaug1.shape[0])
    x0_2 = np.zeros(Aaug2.shape[0])

    # Perform Gauss-Seidel iteration
    solution1 = GaussSeidel(Aaug1, x0_1)
    solution2 = GaussSeidel(Aaug2, x0_2)

    # Print the solutions
    print("Solution for the first system:")
    print(solution1)
    print("\nSolution for the second system:")
    print(solution2)

# Call the main function to execute Gauss-Seidel method
main()
