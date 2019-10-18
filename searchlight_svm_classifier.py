#!/usr/bin/env python

# Arguments: subject number, hemisphere, and what we're training on (taxonomy or behavior)
# Run example: ./searchlight_svm_clf.py 1 r taxonomy
# Erica Busch, Luke Slipski, Marissa Clark

import mvpa2
from os.path import exists, join  
import mvpa2.suite as mv
import glob
import numpy as np
import sys
# from scipy.stats import zscore
# parse arguments 
subid= sys.argv[1]
hemi = sys.argv[2]
train_on=sys.argv[3]

if train_on == 'taxonomy':
	predict='behavior'
else:
	predict='taxonomy'

# format strings to grab all of this subject's hemi's gifti files
sub = '{:0>6}'.format(subid)
data_path = '/dartfs-hpc/scratch/psyc164/mvpaces/glm/'
save_path = '/dartfs-hpc/scratch/psyc164/mvpaces/'
prefix = data_path+'sub-rid'+sub
suffix = hemi+'.coefs.gii'
fn = prefix + '*' + suffix
files = sorted(glob.glob(fn))

# labels for our gifti_dataset 
taxonomy = np.repeat(['bird', 'insect', 'primate', 'reptile', 'unduate'],4)
behavior = np.tile(['eating', 'fighting', 'running', 'swimming'],5)
conditions = [' '.join((beh, tax)) for beh, tax in zip(behavior, taxonomy)]

# load in all of the data into the dataframe
targets = range(1,21)
ds = None
for x in range(len(files)):
	chunks = [x+1]*20
	d = mv.gifti_dataset(files[x], chunks=chunks, targets=targets)
	d.sa['conditions']=conditions
	d.sa['taxonomy'] = taxonomy
	d.sa['behavior'] = behavior
	if ds is None:
		ds = d
	else:      
		ds = mv.vstack((ds,d))
ds.fa['node_indices'] = range(ds.shape[1])
# zscore all of our samples
mv.zscore(ds, chunks_attr='chunks', dtype='float32')
# load in surgace and get searchlight query
radius = 10
surface = mv.surf.read(join(data_path, '{0}.pial.gii'.format(hemi)))
# this is an arbitrary radius and distance metric!
query = mv.SurfaceQueryEngine(surface, radius, distance_metric='dijkstra')
# based off PyMVPA tutorial
clf = mv.LinearNuSVMC(space=predict)

cv = mv.CrossValidation(clf, mv.NFoldPartitioner(attr=train_on),
                     errorfx=lambda p, t: np.mean(p == t),
                     enable_ca=['stats'])
searchlights = mv.Searchlight(cv, queryengine=query, postproc=mv.mean_sample(), roi_ids=None)
sl_clf_results = searchlights(ds)
outstr = save_path+'results/sub'+sub+'_sl_clf_'+predict+'_'+hemi
res = np.array(sl_clf_results)
np.save(outstr, res)









