from logging import logMultiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

TPR = [1/6,2/6,2/6,3/6,4/6,4/6,5/6,1,1,1]
FPR = [0,0,1/4,1/4,1/4,2/4,2/4,2/4,3/4,1]
plt.figure(figsize=(5, 5))
plt.plot(FPR, TPR, marker='.')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC")
plt.savefig("Q5_ROC.png")
plt.show()
