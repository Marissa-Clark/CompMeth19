#!/usr/bin/env python
import sys
from os import mkdir
from os.path import exists, join
import numpy as np
import mvpa2.suite as mv

# base_dir = '/home/nastase/attention/raw_bids'
# scripts_dir = join(base_dir, 'code')
# data_dir = join(base_dir, 'derivatives')
data_dir = '/dartfs-hpc/scratch/psyc164/mvpaces'
glm_dir = join(data_dir, 'glm')
# suma_dir = join(data_dir, 'freesurfer', 'fsaverage6', 'SUMA')
# mvpa_dir = join(data_dir, 'pymvpa')
# if not exists(mvpa_dir):
#     mkdir(mvpa_dir)
savedir = join(data_dir, 'results')

participant = sys.argv[1]
hemi = sys.argv[2]
task = sys.argv[3]
targets = sys.argv[4]
if targets == 'taxonomy':
    chunks='behavior'
elif targets == 'behavior':
    chunks = 'taxonomy'

runs = [1, 2, 3, 4, 5]
n_conditions = 20
n_vertices = 40962
#n_medial = {'lh': 3487, 'rh': 3491}
n_medial = {'lh': 3486, 'rh': 3491}

# Load surface and create searchlight query engine
surf = mv.surf.read(join(glm_dir, '{0}.pial.gii'.format(hemi)))
qe = mv.SurfaceQueryEngine(surf, 10.0, distance_metric='dijkstra')

# Load in surface data sets
dss = []
for run in runs:
    ds = mv.niml.read(join(glm_dir,'sub-{0}_task-{1}_run-{2}_rw-glm.{3}.coefs.niml.dset'.format(
                            participant, task, run, hemi)))
    ds.sa.pop('stats')
    ds.sa['behavior'] = np.tile(['eating', 'fighting', 'running', 'swimming'], 5)
    ds.sa['taxonomy'] = np.repeat(['bird', 'insect', 'primate', 'reptile', 'ungulate'], 4)
    ds.sa['conditions'] = [' '.join((tax, beh)) for tax, beh in zip(ds.sa.taxonomy, ds.sa.behavior)]
    for lab, cond in zip(ds.sa.labels, ds.sa.conditions):
        assert ' '.join(lab.split('#')[0].split('_')) == cond
    ds.sa['runs'] = [run] * 20
    ds.sa['subjects'] = [participant] * 20
    ds.fa['node_indices'] = range(n_vertices)
    dss.append(ds)
ds = mv.vstack(dss)

# Exclude medial wall
medial_wall = np.where(np.sum(ds.samples == 0, axis=0) == n_conditions * 5)[0].tolist()
cortical_vertices = np.where(np.sum(ds.samples == 0, axis=0) < n_conditions * 5)[0].tolist()
assert len(medial_wall) == n_medial[hemi]
assert len(medial_wall) + len(cortical_vertices) == n_vertices

#np.save(join(mvpa_dir, 'cortical_vertices_{0}.npy'.format(hemi)), cortical_vertices)
#cortical_vertices = = np.load(join(mvpa_dir, 'cortical_vertices_{0}.npy').tolist()

# Z-score features across samples
#mv.zscore(ds, chunks_attr='runs')
ds.samples = ((ds.samples - np.mean(ds.samples, axis=1)[:, None])
              / np.std(ds.samples, axis=1)[:, None])

clf = mv.LinearCSVMC(space=targets)

cv = mv.CrossValidation(clf, mv.NFoldPartitioner(attr=chunks), errorfx=mv.mean_match_accuracy)

sl = mv.Searchlight(cv, queryengine=qe, enable_ca=['roi_sizes'],
                    nproc=1, roi_ids=cortical_vertices)
#sl = mv.Searchlight(cv_rsa, queryengine=qe, enable_ca=['roi_sizes'],
#                    nproc=1, results_backend='native', roi_ids=cortical_vertices)
#tmp_prefix='/local/tmp/sam_sl_p{0}_{1}_'.format(participant_id, hemi)
mv.debug.active += ['SLC']
sl_result = sl(ds)

# Average across folds and finalize result on surface
print("Average searchlight size = {0}".format(np.mean(sl.ca.roi_sizes)))

assert sl_result.shape[1] == len(cortical_vertices)
sl_final = np.zeros((1, n_vertices))
np.put(sl_final, cortical_vertices, np.mean(sl_result, axis=0))
assert sl_final.shape == (1, n_vertices)
np.save(join(savedir, 'SAM_search_sub-{0}_task-{1}_cross-clf-{2}.{3}.npy'.format(
                    participant, task, targets, hemi)), np.array(sl_final))
