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
save_path = './healthy_ones/'
save_path2 = './diseases_ones/'
for i in range (0, 340):
	if (df_n[i][0]=='1') :
		filename = './mri' + str(i+1) +'.mhd'
		itkimage_seg = sitk.ReadImage(filename)
		scan = sitk.GetArrayFromImage(itkimage_seg)
		save_file = save_path + 'mri' + str(i+1) +'.mhd'
		sitk.WriteImage(sitk.GetImageFromArray(scan, isVector=False), save_file,False)
	if (df_n[i][0]=='0') :
		filename = './mri' + str(i+1) +'.mhd'
		itkimage_seg = sitk.ReadImage(filename)
		scan = sitk.GetArrayFromImage(itkimage_seg)
		save_file = save_path2 + 'mri' + str(i+1) +'.mhd'
		sitk.WriteImage(sitk.GetImageFromArray(scan, isVector=False), save_file,False)
	# if (i<=290):
	# 	smalldict = {}
	# 	filename = 'mri' + str(i+1) +'.mhd'

	# 	smalldict['image'] = filename
	# 	smalldict ['label'] = df_n[i][0]
	# 	dictout[keyword].append(smalldict)
	# 	print (filename, df_n[i][0])
	# elif (i>=292):
	# 	smalldict = {}
	# 	filename = 'mri' + str(i+1) +'.mhd'
	# 	smalldict['image'] = filename 
	# 	smalldict ['label'] = df_n[i][0]
	# 	dictout[keyword].append(smalldict)
	# 	print (filename, df_n[i][0])

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