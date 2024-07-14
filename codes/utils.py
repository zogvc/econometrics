# Define methods which are used globally in the repository.
# © Seho Jeong. 2024.

# All data for input should be numpy array unless specified.

def remove_outliers(data, limits=(0.05, 0.95)):
    """
    Remove outliers of given array.

    Paramters
    ----------
    data: numpy.ndarray
        Lorem ipsum
    limits: int tuple
        Lorem ipsum
        
    Results
    ----------
    processed_data: numpy.ndarray
        Lorem ipsum
    """
    
    lp, up = limits
    lb = np.percentile(arr, q=lp)
    ub = np.percentile(arr, q=up)

    proccessed_data = data[(arr >= lb) & (arr <= ub)]

    return processed_data