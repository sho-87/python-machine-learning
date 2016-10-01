import matplotlib.pyplot as plt


def plot_training(history):
    """Plot the training curve.
    
    Parameters:
    history -- numpy array/list of cost values over all training iterations
    
    Returns:
    Plot of the cost for each iteration of training
    
    """
    plt.plot(range(1, len(history)+1), history)
    plt.grid(True)
    plt.xlim(1, len(history))
    plt.ylim(min(history), max(history))
    
    plt.title("Training Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")