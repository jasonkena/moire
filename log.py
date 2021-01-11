import numpy as np
import os
from generate import main
import contextlib
from tqdm import tqdm
from itertools import chain

if not os.path.exists('data'):
    os.makedirs('data')

# get the equivalent already calculated value of resistance
# also converts to radians
# expects simplified main
# NOTE NOTE NOTE: remove this memoization function to perform verification
# NOTE if broken, it will rebreak on ValueErrors, but oh well
def memoize(func):
    memo = {}
    def helper(x):
        if (x//60)%2==0:
            equivalent=x%60
        else:
            equivalent=60-(x%60)

        if equivalent not in memo:
            memo[equivalent] = func(np.radians(equivalent))
        return memo[equivalent]
    return helper

def blockPrinting(func):
    def helper(*args, **kwargs):
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            return func(*args, **kwargs)
    return helper

# make only 1 variable in the main function
def simplify(func, config):
    def helper(x):
        return func(config["inner_radius"], config["outer_radius"], x, enable_interlayer=config["enable_interlayer"])
    return helper

def calculate_record(i, func, prefix):
    # to prevent dumb stuff
    try:
        resistance = func(i)
        with open(os.path.join("data",f"{prefix}_data.txt"), "a") as file:
            string = f"{str(i)} {str(resistance.item())}\n"
            file.write(string)
    except Exception as e:
        with open(os.path.join("data",f"{prefix}_errors.txt"), "a") as file:
            string = f"{str(i)}, {str(e)}\n"
            file.write(string)
        pass

# Block print statements
# NOTE NOTE NOTE num_pnts it the number of datapoints between 0 and 60 degrees
keys = ["inner_radius", "outer_radius", "num_pnts", "enable_interlayer"]
values = [10, 30, 60, False]
config =dict(zip(keys, values))

# memoized and blocked printing
final_main = blockPrinting(memoize(simplify(main, config)))

prefix = "_".join(map(str,values))
print(f"Prefix: {prefix}")

iterable = np.linspace(0, 60, endpoint=False, num=config["num_pnts"])

for batch in range(6):
    for i in tqdm(iterable+(60*batch)):
        calculate_record(i, final_main, prefix)

'''
iteration #119 produced error
'''
