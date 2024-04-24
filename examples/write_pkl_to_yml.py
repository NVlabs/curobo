import pickle
import numpy as np
import yaml
import scipy.io as sio

obs_num = 10
datadir = '../no_filter_planning_results/planning_results_pi_6/3d7links'+str(obs_num)+'obs/'
filename = 'armtd_1branched_t0.5_stats_3d7links100trials'+str(obs_num)+'obs150steps_0.5limit.pkl'
# datadir = '../'
# filename = 'sphere_HLP_1branched_t0.5_stats_3d7links14trials12obs150steps_0.5limit.pkl'

worlddir = '../src/curobo/content/configs/world/sparrows_comparison/'+str(obs_num)+'obs/'
# worlddir = '../src/curobo/content/configs/world/sparrows_comparison/hard/'

with open(datadir + filename, 'rb') as f:
    data = pickle.load(f)

for i in range(len(data)):
    file = open(worlddir + 'world_' + str(i) + '.yml', 'w')

    file.write('cuboid:\n')
    for j in range(len(data[i]['initial']['obstacle_pos'])):
        file.write('  obstacle' + str(j) + ':\n')
        file.write('    dims: ' + str(data[i]['initial']['obstacle_size'][j].tolist()) + '\n')
        file.write('    pose: ' + str(np.append(data[i]['initial']['obstacle_pos'][j], np.array([1,0,0,0])).tolist()) + '\n')
    
    file.close()

    sio.savemat(worlddir + 'world_' + str(i) + '.mat', {'obstacle_pos': data[i]['initial']['obstacle_pos'], 'obstacle_size': data[i]['initial']['obstacle_size']})

# for i in range(1, len(data) + 1):
#     file = open(worlddir + 'world_' + str(i) + '.yml', 'w')

#     file.write('cuboid:\n')
#     for j in range(len(data[i]['initial']['obstacle_pos'])):
#         file.write('  obstacle' + str(j) + ':\n')
#         file.write('    dims: ' + str(data[i]['initial']['obstacle_size'][j].tolist()) + '\n')
#         file.write('    pose: ' + str(np.append(data[i]['initial']['obstacle_pos'][j], np.array([1,0,0,0])).tolist()) + '\n')
    
#     file.close()

#     sio.savemat(worlddir + 'world_' + str(i) + '.mat', {'obstacle_pos': data[i]['initial']['obstacle_pos'], 'obstacle_size': data[i]['initial']['obstacle_size']})