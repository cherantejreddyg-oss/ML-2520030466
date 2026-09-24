import pandas as pd
import numpy as np

data = {
    'A' : [1,2,2,4],
    'B' : ['apple','pineapple','apple', np.nan],
    'C' : [10, 10, 10, 10]
}
df = pd .DataFrame(data)

print(df.nunique())
