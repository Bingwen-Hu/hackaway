class Goo(object):
    """
    The Goo object contains lots of spam

    Args:
        arg (str): The arg is used for.

        *args: The variable arguments are used for ...

        **kwargs: The keyword arguments are used for ...

    Attributes:
        arg (str): This is where we store arg.
    """

    def __init__(self, arg, *args, **kwargs):
        self.arg = arg

    def googles(self, amount, cooked):
        """We can't have goo without gle, so here's the gles

        Args:
            amount (int): The amount of eggs to return.

            cooked (bool): Should the eggs be cooked?

        Raises:
            RuntimeError: Out of googles

        Returns:
            Eggs: A bunch o googles
        """
        pass
