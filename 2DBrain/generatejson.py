import SimpleITK as sitk
import os, glob
import json
import numpy as np
import csv

with open('label.csv', newline='') as f:
    reader = csv.reader(f)
    your_list = list(reader)
np.array(your_list)
df_n = np.array(your_list)
keyword = 'train'
dictout = {keyword:[]}
for i in range (0, 301):
	if (i<=290):
		smalldict = {}
		filename = 'mri' + str(i+1) +'.mhd'
	
		smalldict['image'] = filename
		smalldict ['label'] = df_n[i][0]
		dictout[keyword].append(smalldict)
		print (filename, df_n[i][0])
	elif (i>=292):
		smalldict = {}
		filename = 'mri' + str(i+1) +'.mhd'
		smalldict['image'] = filename
		smalldict ['label'] = df_n[i][0]
		dictout[keyword].append(smalldict)
		print (filename, df_n[i][0])

keyword = 'val'
dictout[keyword] = [] 
for i in range (301, len(df_n)-1):
	smalldict = {}
	filename = 'mri' + str(i+1) +'.mhd'
	smalldict['image'] = filename
	smalldict ['label'] = df_n[i][0]
	dictout[keyword].append(smalldict)

savefilename = './data'+ '.json'
with open(savefilename, 'w') as fp:
	json.dump(dictout, fp)