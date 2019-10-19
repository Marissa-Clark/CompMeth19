#!/usr/bin/env python

# Run from ~/attention/raw_bids/code with something like
# ./category_glm_runwise.py rid000001 |& tee ../logs/glm_rid000001_log.txt &

import sys
from os import chdir, makedirs
from os.path import exists, join
from subprocess import call
import numpy as np
from mvpa2.base.hdf5 import h5load

participant = sys.argv[1]
task = sys.argv[2]

data_dir = '/dartfs/rc/lab/D/DBIC/DBIC/datasets.datalad.org/labs/haxby/attention'
target_dir = '/dartfs-hpc/scratch/psyc164/mvpaces'
scripts_dir = join(target_dir, 'code')
func_dir = join(data_dir, 'sub-'+participant, 'func')
fmri_dir = join(data_dir, 'derivatives')
prep_dir = join(fmri_dir, 'fmriprep', 'sub-'+participant, 'func')
glm_dir = target_dir+'/glm'
reg_dir = join(glm_dir, 'regressors', 'runwise')
if not exists(reg_dir):
    makedirs(reg_dir)

# Convert fmriprep's confounds.tsv for 3dDeconvolve -ortvec
ortvecs = []
for run in [1, 2, 3, 4, 5]:
    with open(join(prep_dir,
                   'sub-{0}_task-{1}_run-{2}_bold_confounds.tsv'.format(
                    participant, task, run))) as f:
        lines = [line.strip().split('\t') for line in f.readlines()]

    confounds = {}
    for confound_i, confound in enumerate(lines[0]):
        confound_ts = []
        for tr in lines[1:]:
            confound_ts.append(tr[confound_i])
        confounds[confound] = confound_ts

    keep = ['FramewiseDisplacement', 'aCompCor00', 'aCompCor01', 'aCompCor02',
            'aCompCor03', 'aCompCor04', 'aCompCor05', 'X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
    ortvec = {c: confounds[c] for c in keep}
    ortvecs.append(ortvec)

    # Create de-meaned and first derivatives of head motion (backward difference with leading zero)
    for motion_reg in ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']:
        motion = [float(m) for m in ortvec[motion_reg]]
        motion_demean = np.array(motion) - np.mean(motion)
        motion_deriv = np.insert(np.diff(motion_demean, n=1), 0, 0)
        assert len(motion_demean) == len(motion_deriv) == len(ortvec[motion_reg])
        ortvec[motion_reg + '_demean'] = ['{:.9f}'.format(m) for m in motion_demean]
        ortvec[motion_reg + '_deriv'] = ['{:.9f}'.format(m) for m in motion_deriv]
        del ortvec[motion_reg]
    assert len(ortvec) == 19

    with open(join(reg_dir, 'sub-{0}_task-{1}_run-{2}_bold_ortvec.1D'.format(participant, task, run)), 'w') as f:
        rows = []
        for tr in range(len(ortvec[keep[0]])):
            row = []
            for confound in ortvec:
                if ortvec[confound][tr] == 'n/a':
                    row.append('0')
                else:
                    row.append(ortvec[confound][tr])
            row = '\t'.join(row)
            rows.append(row)
        f.write('\n'.join(rows))

# Reshape BIDS events.tsv to AFNI format
for run in [1, 2, 3, 4, 5]:
    stim_times = {}
    beh_reps = []
    tax_reps = []
    buttons = []
    with open(join(func_dir, 'sub-{0}_task-{1}_run-{2}_events.tsv'.format(participant, task, run)), 'r') as f:
        events = [line.strip().split('\t') for line in f.readlines()]
    for event in events[1:]:
        if event[7] != 'none':
            buttons.append('{0:.3f}'.format(float(event[0]) + float(event[7])))
        if event[6] == 'behavior':
            beh_reps.append(event[0])
        elif event[6] == 'taxonomy':
            tax_reps.append(event[0])
        else:
            if event[2] not in stim_times:
                stim_times[event[2]] = []
            stim_times[event[2]].append(event[0])
    assert len(stim_times) == 20
    with open(join(reg_dir, 'sub-{0}_task-{1}_run-{2}_beh_repetitions.txt'.format(participant, task, run)), 'w') as f:
        f.write(' '.join(beh_reps))
    with open(join(reg_dir, 'sub-{0}_task-{1}_run-{2}_tax_repetitions.txt'.format(participant, task, run)), 'w') as f:
        f.write(' '.join(tax_reps))
    with open(join(reg_dir, 'sub-{0}_task-{1}_run-{2}_buttons.txt'.format(participant, task, run)), 'w') as f:
        f.write(' '.join(buttons))
    for category, onsets in stim_times.items():
        with open(join(reg_dir, 'sub-{0}_task-{1}_run-{2}_{3}.txt'.format(participant, task, run, category)), 'w') as f:
            f.write(' '.join(onsets))
        
# Change directory for AFNI
chdir(glm_dir)

# Run AFNI's 3dDeconvolve
for run in [1, 2, 3, 4, 5]:

    # Set up regressors for 3dDeconvolve
    stim_regs = []
    for stim_i, stim_label in enumerate(sorted(stim_times.keys())):
        stim_regs.append("-stim_times {0} {1} 'BLOCK(2.0,1)' -stim_label {0} {2}".format(
                stim_i + 1, join(reg_dir, 'sub-{0}_task-{1}_run-{2}_{3}.txt'.format(participant, task, run, stim_label)), stim_label))

    beh_rep_reg = "-stim_times 21 {0} 'BLOCK(2.0,1)' -stim_label 21 behavior_repetition".format(
            join(reg_dir, 'sub-{0}_task-{1}_run-{2}_beh_repetitions.txt'.format(participant, task, run)))

    tax_rep_reg = "-stim_times 22 {0} 'BLOCK(2.0,1)' -stim_label 22 taxonomy_repetition".format(
            join(reg_dir, 'sub-{0}_task-{1}_run-{2}_tax_repetitions.txt'.format(participant, task, run)))

    buttons_reg = "-stim_times 23 {0} 'BLOCK(1.0,1)' -stim_label 23 button_press".format(
            join(reg_dir, 'sub-{0}_task-{1}_run-{2}_buttons.txt'.format(participant, task, run)))

    for side, hemi in [('L', 'lh'), ('R', 'rh')]:
        cmd = ("3dDeconvolve -polort A -jobs 8 -local_times -force_TR 2.0 "
                    "-input "
                    "{0}/sub-{1}_task-{2}_run-{3}_bold_space-fsaverage6.{4}.func.gii "
                    "-num_stimts 23 ".format(prep_dir, participant, task, run, side) +
                    ' '.join(stim_regs) + ' ' + ' '.join((beh_rep_reg, tax_rep_reg, buttons_reg)) + ' ' +
                    "-ortvec {5} "
                    "-fout -tout -x1D {0}/sub-{1}_task-{2}_run-{3}_rw-glm.{4}.X.xmat.1D "
                    "-xjpeg {0}/sub-{1}_task-{2}_run-{3}_rw-glm.{4}.X.jpg "
                    "-fitts {0}/sub-{1}_task-{2}_run-{3}_rw-glm.{4}.fitts "
                    "-errts {0}/sub-{1}_task-{2}_run-{3}_rw-glm.{4}.errts "
                    "-bucket {0}/sub-{1}_task-{2}_run-{3}_rw-glm.{4}.stats".format(
                        glm_dir, participant, task, run, hemi, join(reg_dir,
                'sub-{0}_task-{1}_run-{2}_bold_ortvec.1D'.format(participant, task, run))))
        call(cmd, shell=True)

        # Run AFNI's 3dREMLfit
        chdir(glm_dir)
        with open(join(glm_dir, 'sub-{0}_task-{1}_run-{2}_rw-glm.REML_cmd'.format(participant, task, run))) as f: 
            reml_cmd = '3dREMLfit' + f.read().split('3dREMLfit')[-1]
        call(reml_cmd, shell=True)

        # Grab coefficients
        chdir(glm_dir)
        call("3dbucket -prefix sub-{0}_task-{1}_run-{2}_rw-glm.{3}.coefs.gii "
             "'sub-{0}_task-{1}_run-{2}_rw-glm.{3}.stats_REML.gii[1..39(2)]'".format(participant, task, run, hemi), shell=True)

        call("ConvertDset -o_niml_asc -input sub-{0}_task-{1}_run-{2}_rw-glm.{3}.coefs.gii "
             "-prefix sub-{0}_task-{1}_run-{2}_rw-glm.{3}.coefs".format(participant, task, run, hemi), shell=True)
