import numpy as np

noises = ['glass_click', 'knock', 'frying_pan', 'soccer_ball', 'mug', 'saxophone', 'alarm', 'bang', 'beep', 'roomba']
solvers = ['tsp', 'info_max', 'emmd']
cases = ['no_noise', 'light_noise', 'strong_noise']
rooms = ['drone_room_1', 'drone_room_2', 'drone_room_3']

print('Trajectory Data ---------------------------------------------------------------------------')
for room in rooms: 
    print(f'Room: {room} --------------------------')
    for solver in solvers:
        numero = 0
        covg = np.zeros(len(noises)*len(cases))
        distance = np.zeros(len(noises)*len(cases))
        for case in cases:
            for noise in noises:
                output = np.load(f'{room}/{solver}/{case}/{noise}/output.npz')
                covg[numero] = output['visitation_per_distance']
                distance[numero] = output['distance']
                numero += 1
        print(f'{solver} coverage: {np.mean(covg)}')
        print(f'{solver} distance: {np.mean(distance)}')
    print('\n')

print('\n')

print('Sound Accuracy Data -----------------------------------------------------------------------')

output = np.load('cumulative_results.npy')

args = {'emmd': {'no_noise': {}, 'light_noise': {}, 'strong_noise': {}}, 
        'tsp': {'no_noise': {}, 'light_noise': {}, 'strong_noise': {}}, 
        'info_max': {'no_noise': {}, 'light_noise': {}, 'strong_noise': {}}}
sound_args = []

for i in range(len(output)):
    if output[i][2] not in sound_args:
        np.append(sound_args, str(output[i][2]))

    if output[i][2] not in args[str(output[i][0])][str(output[i][1])]:
        args[str(output[i][0])][str(output[i][1])][str(output[i][2])] = np.array([[str(output[i][3]), output[i][-1]]])
    else:
        args[str(output[i][0])][str(output[i][1])][str(output[i][2])] = np.append(args[str(output[i][0])][str(output[i][1])][str(output[i][2])], np.array([[str(output[i][3]), output[i][-1]]]), axis=0)

use_sounds = ['glass_click', 'bang', 'beep']

print('\n')

"""
for sound in use_sounds:
    print(f'Sound: {sound} --------------------------')
    for case in cases:
        if case == 'strong_noise':
            continue
        else:
            print(f'Case: {case}')
            print('\n')
            for solver in solvers:
                print(f'Solver: {solver}')
                for item in range(len(args[solver][case][sound])):
                    print(f'Item: {args[solver][case][sound][item][0]}')
                    print(f'Confidence: {args[solver][case][sound][item][1]}')
                print('\n')
            print('\n')
            """

for solver in solvers:
    print(f'Solver: {solver} --------------------------')
    for sound in use_sounds:
        print(f'Sound: {sound}')
        print('\n')
        for case in cases:
            if case == 'strong_noise':
                continue
            else:
                print(f'Case: {case}')
                for item in range(len(args[solver][case][sound])):
                    print(f'Item: {args[solver][case][sound][item][0]}')
                    print(f'Confidence: {args[solver][case][sound][item][1]}')
            print('\n')
        print('\n')


print('Missing Sounds -----------------------------------------------------------------------------')
count = 0
missing_sounds = []

for solver in solvers:    
    for case in cases:
        for noise in noises:
            try:
                args[solver][case][noise]
            except:
                if noise not in missing_sounds:
                    missing_sounds.append(noise)

print(missing_sounds)

