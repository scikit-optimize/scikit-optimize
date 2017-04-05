"""Stopping criteria."""


def delta_x(delta):
    def delta_x(next_x, result):
        if result.space.distance(next_x, result.x_iters[-1]) < delta:
            return True
        else:
            return False

    return delta_x
