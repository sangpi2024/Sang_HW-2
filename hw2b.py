import numpy as np

def Secant(fcn, x0, x1, maxiter=10, xtol=1e-5):
    """
    Use the Secant Method to find the root of fcn(x), in the neighborhood of x0 and x1.
    Parameters:
    - fcn: the function for which we want to find the root
    - x0 and x1: two x values in the neighborhood of the root
    - xtol: exit if the |xnewest - xprevious| < xtol
    - maxiter: exit if the number of iterations (new x values) equals this number

    Returns:
    - The final estimate of the root (most recent new x value)
    """
    x_prev = x0
    x_curr = x1

    for i in range(maxiter):
        f_prev = fcn(x_prev)
        f_curr = fcn(x_curr)

        # Avoid division by zero
        if f_curr - f_prev == 0:
            break

        x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)

        # Check convergence
        if abs(x_new - x_curr) < xtol:
            return x_new

        x_prev, x_curr = x_curr, x_new

    return x_curr

def main():
    # Equation 1: x - 3*cos(x) = 0
    fcn1 = lambda x: x - 3 * np.cos(x)
    root1 = Secant(fcn1, x0=1, x1=2, maxiter=5, xtol=1e-4)
    print("Equation 1 Root:", root1)

    # Equation 2: cos(2*x)*x^3 = 0
    fcn2 = lambda x: np.cos(2 * x) * x ** 3
    root2 = Secant(fcn2, x0=1, x1=2, maxiter=15, xtol=1e-8)
    print("Equation 2 Root:", root2)

    # Equation 3: cos(2*x)*x^3 = 0 with limited iterations
    fcn3 = lambda x: np.cos(2 * x) * x ** 3
    root3 = Secant(fcn3, x0=1, x1=2, maxiter=3, xtol=1e-8)
    print("Equation 3 Root:", root3)


if __name__ == "__main__":
    main()
