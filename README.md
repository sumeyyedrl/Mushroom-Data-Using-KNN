# How to Use the Mushroom Data
To access the data you can click on the [link](https://archive.ics.uci.edu/dataset/73/mushroom)
To use the data there is no need to download it. First ucimlrepo package needs to be installed; 
```
pip install ucimlrepo
```
Then the mushroom data can be used simply by adding these code to the pyhton code; 
```
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
mushroom = fetch_ucirepo(id=73) 
  
# data (as pandas dataframes) 
X = mushroom.data.features 
y = mushroom.data.targets 
  
# metadata 
print(mushroom.metadata) 
  
# variable information 
print(mushroom.variables) 
```
