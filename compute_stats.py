import numpy as np
import sys
import glob
import os
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score 



gt_folder = sys.argv[1] 
result_folder = sys.argv[2]

kappa = []
acc = []
f1 = []
pc_f1 = []


for i in range(5):
	print (i+1)
	gtFName = gt_folder+"/test_gt_"+str(i+1)+".npy"
	clFName = result_folder+"/pred_"+str(i+1)+".npy"
	if os.path.exists(clFName):
		gt = np.load(gtFName)
		gt = gt[:,1] 
		cl = np.load(clFName)
		kappa.append( cohen_kappa_score(gt,cl) )
		print (f1_score(gt,cl, average="weighted") )
		f1.append( f1_score(gt,cl, average="weighted") * 100 )
		acc.append( accuracy_score(gt,cl) * 100 )
		pc_f1.append( f1_score(gt,cl, average=None) * 100 )
		print (f1_score(gt,cl, average=None) )


pc_f1  = np.vstack(pc_f1)

print ("F1:",np.mean(f1),np.std(f1))
print ("Kappa:",np.mean(kappa),np.std(kappa))
print ("Acc:",np.mean(acc),np.std(acc),"\n")



print ( f'{"%.2f"%round(np.mean(f1),2)} $\\pm$ {"%.2f"%round(np.std(f1),2)} & {"%.3f"%round(np.mean(kappa),3)} $\\pm$ {"%.3f"%round(np.std(kappa),3)} & {"%.2f"%round(np.mean(acc),2)} $\\pm$ {"%.2f"%round(np.std(acc),2)} \\\\' )
seq = [("%.2f"%a+" $\\pm$ "+"%.2f"%b) for (a,b) in np.column_stack([np.mean(pc_f1,axis=0),np.std(pc_f1,axis=0)]) ]
print (' & '.join(el.split(" $\\pm$ ")[0] for el in seq) )
print (' & '.join(seq) )
# seq1 = [("(%.2f"%a+','+"%.2f)"%b) for (a,b) in np.column_stack([np.mean(pc_f1,axis=0),np.std(pc_f1,axis=0)]) ]
# print (seq1)


