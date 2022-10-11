import json
import random 
oridata = json.load ((open('./train_atlas.json', 'r')))
# print (oridata['trainatlas'])
oridata['testatlas'] = {}
for i in range (0, 40):
    x = random.randint(0,len(oridata['trainatlas']))
    print(i)
    oridata['testatlas'][i] = oridata['trainatlas'][x]
    del oridata['trainatlas'][x]

savefilename = './test'+ '.json'
with open(savefilename, 'w') as fp:
	json.dump(oridata, fp)

