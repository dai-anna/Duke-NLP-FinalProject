#%%
import pandas as pd


# get second largest number from list
def get_second_largest(numbers):
    m1, m2 = 0, 0
    for x in numbers:
        if x > m1:
            m1, m2 = x, m1
        elif x > m2:
            m2 = x
    return m2