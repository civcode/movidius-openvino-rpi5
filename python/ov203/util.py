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

def note(message):
    import sys
    print(message,file=sys.stderr,flush=True)

def windowed_rate(stamps,n=50):
    if len(stamps)<2: return 0.0
    tail=stamps[-(n+1):]; span=tail[-1]-tail[0]
    return (len(tail)-1)/span if span>0 else 0.0
