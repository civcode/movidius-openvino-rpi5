import argparse

def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number

def non_negative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return number

def positive_float(value):
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number

def non_negative_float(value):
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return number

def probability(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number
