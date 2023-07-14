import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def MAP():

    # Generate some sample data
    np.random.seed(123)
    sample_data = np.random.normal(loc=5, scale=2, size=100)

    # Define the prior distribution parameters
    prior_mean = 0
    prior_std = 5

    # Calculate the posterior distribution parameters
    posterior_mean = (prior_std**2 * np.mean(sample_data) + np.var(sample_data) * prior_mean) / (np.var(sample_data) + prior_std**2)
    posterior_std = np.sqrt((np.var(sample_data) * prior_std**2) / (np.var(sample_data) + prior_std**2))

    # Plot the sample data histogram
    plt.hist(sample_data, bins=20, density=True, alpha=0.5, label='Sample Data')

    # Plot the prior distribution
    x = np.linspace(-10, 20, 100)
    prior_distribution = norm.pdf(x, prior_mean, prior_std)
    plt.plot(x, prior_distribution, 'r-', label='Prior Distribution')

    # Plot the posterior distribution
    posterior_distribution = norm.pdf(x, posterior_mean, posterior_std)
    plt.plot(x, posterior_distribution, 'g-', label='Posterior Distribution')

    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.title('Maximum a Posteriori Estimation')
    plt.legend()
    plt.savefig("plots/MAP.jpg", dpi=400)

def map_coin():
    # Define the parameters
    h = 6     # Number of heads
    t = 4     # Number of tails
    α = 2     # Parameter alpha for the beta distribution
    β = 2     # Parameter beta for the beta distribution

    # Define the range of p values
    p = np.linspace(0.01, 0.99, 100)

    # Calculate the prior and posterior
    prior = beta.pdf(p, α, β)
    posterior = beta.pdf(p, h + α, t + β)

    # Calculate MAP
    p_MAP = (h + α - 1) / (h + t + α + β - 2)

    # Plot the distributions
    plt.figure(figsize=(10, 7))
    plt.plot(p, prior, label='Prior')
    plt.plot(p, posterior, label='Posterior')
    plt.axvline(p_MAP, color='red', linestyle='dotted', label='MAP')
    plt.legend(loc='upper left')
    plt.xlabel('p')
    plt.ylabel('Density')
    plt.title('MAP Estimation for Coin Bias')
    plt.savefig("plots/MAP_coin.jpg", dpi=400)

def MLE():
    import numpy as np
    from scipy.stats import norm
    import matplotlib.pyplot as plt

    # Randomly generate a data sample
    np.random.seed(0)
    sample_data = np.random.normal(loc=5.0, scale=2.0, size=1000)

    # Calculate the MLE estimates
    mu_mle = np.mean(sample_data)
    sigma_mle = np.std(sample_data)

    # Generate points for plotting
    x = np.linspace(np.min(sample_data), np.max(sample_data), 1000)
    y = norm.pdf(x, loc=mu_mle, scale=sigma_mle)

    # Plot the sample data and fitted Gaussian
    plt.figure(figsize=(10, 6))
    plt.hist(sample_data, bins=100, density=True, alpha=0.6, color='g')
    plt.plot(x, y, 'r', alpha=0.6, label=f'Fit results: mu = {mu_mle:.2f},  sigma = {sigma_mle:.2f}')
    plt.xlabel('Data values')
    plt.ylabel('Density')
    plt.title('Maximum Likelihood Estimation (MLE) of Gaussian parameters')
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig("plots/MLE.jpg", dpi=400)

map_coin()
