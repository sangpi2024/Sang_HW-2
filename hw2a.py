import numpy as np

def Probability(PDF, args, c, GT=True):
    μ, σ = args

    # Define the integration limits
    if GT:
        lower_limit = μ - 5 * σ
        upper_limit = c
    else:
        lower_limit = μ - 5 * σ
        upper_limit = c

    # Use Simpson's 1/3 rule for numerical integration
    n = 1000  # Number of intervals for integration
    h = (upper_limit - lower_limit) / n

    x_values = np.linspace(lower_limit, upper_limit, n + 1)
    f_values = PDF((x_values, μ, σ))
    integral = h / 3 * (f_values[0] + 4 * sum(f_values[1:-1:2]) + 2 * sum(f_values[2:-2:2]) + f_values[-1])
    return integral

def normal_pdf(args):
    x, μ, σ = args
    return 1 / (σ * np.sqrt(2 * np.pi)) * np.exp(-(x - μ) ** 2 / (2 * σ ** 2))

def main():
    # Find P(x<105|N(100,12.5))
    result1 = Probability(normal_pdf, (100, 12.5), 105, GT=False)
    print(f'P(x<105|N(100,12.5))={result1:.2f}')

    # Find P(x>μ+2σ|N(100, 3))
    result2 = Probability(normal_pdf, (100, 3), 100 + 2 * 3, GT=True)
    print(f'P(x>{100 + 2 * 3}|N(100,3))={result2:.2f}')

if __name__ == "__main__":
    main()
