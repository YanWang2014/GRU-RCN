import random
import numpy as np

random.seed(42)

print(random.sample(range(20),k=10))
print(np.random.uniform(1,1000))

st = random.getstate()  
print(random.sample(range(20),k=20)) 
print(np.random.uniform(1,1000))

random.setstate(st)     
print(random.sample(range(20),k=10)) 
print(np.random.uniform(1,1000))

random.setstate(st)     
print(random.sample(range(20),k=10)) 
print(random.sample(range(20),k=10)) 
print(np.random.uniform(1,1000))