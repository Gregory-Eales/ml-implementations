import random
import numpy as np
import time


def check_parity(bit_string):
    ones = 0
    zeros = 0
    for i in range(len(bit_string)):
        if bit_string[i] == 0:
            zeros+=1
        else:
            ones+=1
    if ones % 2 == 0:
        return True

    else: return False

def check_parity_numpy(bit_string):
    s = np.sum(bit_string)
    if s%2 == 0:
        return True
    else:
        return False

# generate bit string of length N
def generate_bit_string(n):
    bit_string = []
    for i in range(n):
        bit_string.append(random.randint(0, 1))

    parity = check_parity(bit_string)

    return bit_string, parity

def generate_bit_strings(m, n):
    bit_strings = []
    parities = []
    for i in range(m):
        bit_string, parity = generate_bit_string(n)
        bit_strings.append(bit_string)
        parities.append(parity)
    return bit_strings, parities


def main():
    t = time.time()
    bit_strings, parities = generate_bit_strings(100000, 50)
    print(time.time()-t)

if __name__ == "__main__":
    main()
