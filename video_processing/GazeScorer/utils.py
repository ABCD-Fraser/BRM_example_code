import os
import logging


def describe(x, description='', level='brief'):
    """
    Print out information about a variable.
    Args:
        x (any): the variable to describe
        description (string): text to describe the variable - default empty
        level (string): description detail level - 'values', 'brief' (default)
    Returns:
        none
    """

    if isinstance(x, list):
        if description:
            print(description)
        print('list with ', len(x), ' ', type(x[0]), ' elements')
        if(level == 'values'):
            print('Values: \n{}'.format(x))
    else:
        if description:
            print(description)
        print(x.shape, ' array')
        if(level == 'values'):
            print('Values: \n{}'.format(x))


def create_folder(path, overwrite=True, verbose=True):
    """
        Creates a folder if it does not exist.

        Args:
            path (string): path to required folder
            verbose (bool): print out info or warnings (default: True)
        Returns:
            path (string): the path or '' deoending on status
            status (bool): whether the action could be completed or not
    """

    logger = logging.getLogger(__name__)

    if overwrite:
        if os.path.isdir(path):
            if verbose:
                logger.warning(
                    f'Folder {path} already exists, content will be overwritten')
            os.makedirs(path, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)
    else:
        if os.path.isdir(path):
            if verbose:
                logger.warning(f'Folder {path} already exists, skipping')
        else:
            os.makedirs(path, exist_ok=True)

    return path, True
