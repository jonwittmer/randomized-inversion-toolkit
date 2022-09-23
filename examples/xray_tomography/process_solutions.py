import numpy as np
import os
import sys
sys.path.insert(0, os.path.realpath('../..'))
from utils.generate_tables import writeLatexTables

def getFilenames(filenames):
    # remove solutions with different mesh size
    split_filenames = [name.split('_') for name in files]
    for split_name in split_filenames:
        print(split_name[-1].split('.')[0])
        full_name = '_'.join(split_name)
        split_name.append(full_name)
        print(split_filenames)
    return split_filenames

def getInfoFromName(split_name):
    info = {
        'filename' : split_name[-1],
        'samples' : int(split_name[1].split('.')[0][1:]),
        'method' : split_name[0]
    }
    return info

def computeRelativeError(filename, target_solution):
    # compute the relative error with respect to target_solution
    test_solution = np.load('solutions/xray_tomography_gaussian/' + filename)
    return np.linalg.norm(test_solution - target_solution) / np.linalg.norm(target_solution)

def sortBySamples(result):
    zipped_results = zip(result['samples'], result['rel_error'])
    sorted_results = sorted(zipped_results, key=lambda pair: pair[0])
    result['samples'] = [x for x, _ in sorted_results]
    result['rel_error'] = [x for _, x in sorted_results]
    #print(result)
    #print()
    return result

def prettyPrint(results):
    for method in results:
        print(method, results[method])
    return

mesh_size = 4225
u1_solution_filename = 'u1_N00000.npy'.format(mesh_size)
files = os.listdir('solutions/xray_tomography_gaussian/')
print(files)
print(u1_solution_filename)
files.remove(u1_solution_filename)
split_filenames = getFilenames(files)

u1_solution = np.load('solutions/xray_tomography_gaussian/' + u1_solution_filename)
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
    results[info['method']] = sortBySamples(results[info['method']])
    prettyPrint(results)
    print()
print(results)

writeLatexTables(results, 'xray_tomography_gaussian.tex')
