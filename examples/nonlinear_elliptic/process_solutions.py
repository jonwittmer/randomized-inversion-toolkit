import numpy as np
import os
import sys
sys.path.insert(0, os.path.realpath('../..'))
from utils.generate_tables import writeLatexTables

def getFilenamesByMeshSize(filenames, mesh_size):
    # remove solutions with different mesh size
    split_filenames = [name.split('_') for name in files]
    for split_name in split_filenames:
        print(split_name[-1].split('.')[0])
        if split_name[-1].split('.')[0] != str(mesh_size):
            split_filenames.remove(split_name)
        else:
            full_name = '_'.join(split_name)
            split_name.append(full_name)
        print(split_filenames)
    return split_filenames

def getInfoFromName(split_name):
    info = {
        'filename' : split_name[-1],
        'mesh_size' : int(split_name[-2].split('.')[0]),
        'samples' : int(split_name[-4]),
        'method' : '_'.join(split_name[:-4])
    }
    return info

def computeRelativeError(filename, target_solution):
    # compute the relative error with respect to target_solution
    test_solution = np.load('solutions/' + filename)
    return np.linalg.norm(test_solution - target_solution) / np.linalg.norm(target_solution)
    
mesh_size = 1089
u1_solution_filename = 'no_randomization_0_mesh_{}.npy'.format(mesh_size)
files = os.listdir('solutions')
print(files)
print(u1_solution_filename)
files.remove(u1_solution_filename)
split_filenames = getFilenamesByMeshSize(files, mesh_size)

u1_solution = np.load('solutions/' + u1_solution_filename)
results = {}
for name in split_filenames:
    info = getInfoFromName(name)
    info['rel_error'] = computeRelativeError(info['filename'], u1_solution)
    if info['method'] in results:
        results[info['method']]['samples'].append(info['samples'])
        results[info['method']]['rel_error'].append(info['rel_error'])
    else:
        results[info['method']] = {
            'samples' : [info['samples']],
            'rel_error' : [info['rel_error']]
        }
print(results)

writeLatexTables(results, 'nonlinear_elliptic.tex')
