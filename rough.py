cz=4
import numpy as np

data=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
data2=np.arange(0,20,1)
print(data2)
for i in range(0,len(data2)-cz,1):

    input=data2[i:i+cz]
    output=data2[i+cz]
    print(f"Input : {input} -- Output : {output} \n")

