print("Laser-AI")

def closest(x, values):
    """
    This function will return the closest value/ number to a given value
    :param x: (random) value that is to be inspected
    :param values: list of values x is to be compared to
    :return: value closest to x
    """

    #TODO: This function need to be added to the actual network. More in depth discussion about in the next meeting.

    for i, value in enumerate(values):

        if i == 0:
            closest = value
            p_diff = abs(value-x)   # p_diff is the closest difference measured yet
            continue

        diff = abs(value - x)   # diff is the difference of the current value to x

        if diff < p_diff:
            closest = value
            p_diff = diff

    return closest
