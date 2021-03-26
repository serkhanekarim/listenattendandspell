#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from model.listen import Listen

def main():
    """
    Will display the Listen Encoder model
    """
    x = np.random.rand(10,5)
    Listen(x)

if __name__ == "__main__":
    main()