import os
import shutil

import math
import random
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import configparser as CP
import torchaudio.transforms

import pyroomacoustics as pra

import torch
torch.set_num_threads(8)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["OMP_NUM_THREADS"] = '4'
os.environ["OPENBLAS_NUM_THREADS"] = '4'
os.environ["MKL_NUM_THREADS"] =  '4'
os.environ["VECLIB_MAXIMUM_THREADS"] = '4'
os.environ["NUMEXPR_NUM_THREADS"] = '4'



def createDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def deleteDirectory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)


def getParams(args):
    params = dict()
    params['args'] = args
    cfgpath = os.path.join(os.path.dirname(__file__), args.cfg)
    assert os.path.exists(cfgpath), f'No configuration file as [{cfgpath}]'

    cfg = CP.ConfigParser()
    cfg._interpolation = CP.ExtendedInterpolation()
    cfg.read(cfgpath, encoding='UTF-8')
    params['cfg'] = cfg._sections[args.cfg_str]
    cfg = params['cfg']

    params['fs'] = int(cfg['sampling_rate'])

    params['room_rt60_min'] = float(cfg['room_rt60_min'])
    params['room_rt60_max'] = float(cfg['room_rt60_max'])

    params['room_width_min'] = float(cfg['room_width_min'])
    params['room_width_max'] = float(cfg['room_width_max'])
    params['room_length_min'] = float(cfg['room_length_min'])
    params['room_length_max'] = float(cfg['room_length_max'])
    params['room_height_min'] = float(cfg['room_height_min'])
    params['room_height_max'] = float(cfg['room_height_max'])

    params['room_offset_inside'] = float(cfg['room_offset_inside'])

    params['microphone_num'] = int(cfg['microphone_num'])
    params['microphone_radius'] = float(cfg['microphone_radius'])

    params['array_source_distance_min'] = float(cfg['array_source_distance_min'])
    params['array_source_distance_max'] = float(cfg['array_source_distance_max'])

    params['angle_between_sources_min'] = float(cfg['angle_between_sources_min'])

    params['rir_proc_dir'] = str(cfg['rir_proc_dir'])

    return params


def plot3dRoom(walls, sources, mic_array, figsize=None, mic_marker_size=10, ax=None):
    """
    Refer to https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/room.py#L1464
    Plots the room with its walls, microphones, sources and images
    Difference with the above link is to change ax.scatter to ax.scatter3D
    """

    fig = None

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = a3.Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

    # plot the walls
    for w in walls:
        tri = a3.art3d.Poly3DCollection([w.corners.T], alpha=0.5)
        tri.set_color(colors.rgb2hex(np.random.rand(3)))
        tri.set_edgecolor("k")
        ax.add_collection3d(tri)

    # define some markers for different sources and colormap for damping
    cmap = plt.get_cmap("YlGnBu")

    # plot the sources
    for i, source in enumerate(sources):
        ax.scatter3D(source[0], source[1], source[2],
                     c=[cmap(1.0)], s=20, marker="s", edgecolor=cmap(1.0))

    # plot the microphones
    for i, mic in enumerate(mic_array):
        ax.scatter3D(mic[0], mic[1], mic[2],
                     c="k", s=mic_marker_size, marker="+", linewidth=0.5)

    return fig, ax


def saveFigure(params, room, rt60, room_geometry, array_center_location, mic_angle, source_location_list, save_dir):
    width, length, height = room_geometry[0], room_geometry[1], room_geometry[2]
    radius = params['microphone_radius']


    sources = []
    xs, ys, zs = [], [], []
    for i in range(len(source_location_list)):
        _xs = source_location_list[i][0]
        _ys = source_location_list[i][1]
        _zs = source_location_list[i][2]
        sources.append((_xs, _ys, _zs))
        xs.append(_xs); ys.append(_ys); zs.append(_zs)

    mic_array = []
    for i in range(params['microphone_num']):
        _xm = array_center_location[0] + radius*math.cos(mic_angle + math.pi*2/params['microphone_num']*i)
        _ym = array_center_location[1] + radius*math.sin(mic_angle + math.pi*2/params['microphone_num']*i)
        _zm = array_center_location[2]
        mic_array.append((_xm, _ym, _zm))


    # 3D plot
    plt.figure(1)
    fig, ax = plot3dRoom(room.walls, sources, mic_array)

    ax.set_xlim([-1, max(width, length)+1]); ax.set_ylim([-1, max(width, length)+1]); ax.set_zlim([-1, height+1])
    ax.set_xlabel('x-axis'); ax.set_ylabel('y-axis'); ax.set_zlabel('z-axis')

    ax.text(array_center_location[0], array_center_location[1], array_center_location[2], 'M')
    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        label = 'S%d' % (i+1)
        ax.text(x, y, z, label)

    plt.savefig(save_dir+'.png', dpi=300)
    plt.close()


    # 3D plot (top view)
    plt.figure(2)
    fig, ax = plot3dRoom(room.walls, sources, mic_array)

    ax.view_init(elev=90, azim=0)

    ax.set_xlim([-1, max(width, length)+1]); ax.set_ylim([-1, max(width, length)+1]);  ax.set_zlim([-1, height+1])
    ax.set_xlabel('x-axis'); ax.set_ylabel('y-axis'); ax.set_zlabel('z-axis')

    ax.text(array_center_location[0], array_center_location[1], array_center_location[2], 'M')
    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        label = 'S%d' % (i+1)
        ax.text(x, y, z, label)

    plt.savefig(save_dir+'(top).png', dpi=300)
    plt.close()


    # Write the information on .txt file
    array_center_location = '(%.2f,%.2f,%.2f)' % (array_center_location[0],
                                                  array_center_location[1],
                                                  array_center_location[2])
    mic_angle = mic_angle * 180 / math.pi

    f = open(f'{save_dir}.txt', 'w')
    f.write(f'[Room]\n'
            f'length: {length:.2f}m\n'
            f'width: {width:.2f}m\n'
            f'height: {height:.2f}m\n'
            f'RT60: {rt60:.2f}s\n\n')

    f.write(f'[Mic array]\n'
            f'center: {array_center_location}m\n'
            f'radius: {radius:.2f}m\n'
            f'rot angle: {mic_angle:.2f}deg\n\n')

    f.write(f'[Sources]\n')
    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        label = '%d: (%.2f,%.2f,%.2f)m\n' % (i+1, x, y, z)
        f.write(label)
    f.close()


def generateRIRs(params, foreground_list, file):
    # Generate RIRs for each mixture (jams) to make reverberant mixture and target --> rirs1
    # Making target as direct wave (dereverberation) is future work --> rirs2

    # while True:
    np.random.seed(random.randint(0, 65536))
    max_tries_room = 5000

    # Set RT60
    rt60 = params['room_rt60_min'] + np.random.rand() * (params['room_rt60_max'] - params['room_rt60_min'])


    # Set room geometry
    width = params['room_width_min'] + np.random.rand() * (params['room_width_max'] - params['room_width_min'])
    length = params['room_length_min'] + np.random.rand() * (params['room_length_max'] - params['room_length_min'])
    height = params['room_height_min'] + np.random.rand() * (params['room_height_max'] - params['room_height_min'])
    room_geometry = np.array([width, length, height])

    # Build shoebox room
    # Set absorption coefficients
    e_absorption, _ = pra.inverse_sabine(rt60=rt60, room_dim=room_geometry)
    # reverb (Reflect direct sound and early reflection (until 6-th order))
    room1 = pra.ShoeBox(room_geometry, fs=params['fs'], max_order=6, ray_tracing=True,
                        materials=pra.Material(e_absorption))


    # Set microphone array location
    axx = width / 2 + (-0.5) + np.random.rand() * 1   # 1 = 0.5 - (-0.5)
    ayy = length / 2 + (-0.5) + np.random.rand() * 1  # 1 = 0.5 - (-0.5)
    azz = 1 + np.random.rand() * 1                    # 1 = 2 - 1
    array_center_location = np.array([axx, ayy, azz])
    mic_angle = np.random.rand() * math.pi / 4

    # Add microphone array
    mics = pra.beamforming.circular_microphone_array_xyplane(center=array_center_location,
                                                             M=params['microphone_num'], phi0=mic_angle,
                                                             radius=params['microphone_radius'],
                                                             fs=params['fs'], directivity=None, ax=None)
    # reverb
    room1.add(mics)


    # Set speech source"s" location
    source_location_list = []
    for i in range(len(foreground_list)):
        if i == 0:
            cnt = 0
            while True:
                source_location = params['room_offset_inside'] + \
                                  np.random.rand(3) * (room_geometry - params['room_offset_inside'] * 2)
                #######################################################
                if params['rir_proc_dir'].split('_')[2] == 'same':
                    source_location[-1] = azz
                #######################################################
                src_dist = np.sqrt(np.sum(np.power(source_location - array_center_location, 2)))

                if params['array_source_distance_min'] < src_dist < params['array_source_distance_max']:
                    break
                cnt += 1
                if cnt > max_tries_room:
                    assert 0, f"Speech source locating failed."
        else:
            cnt = 0
            while True:
                source_location = params['room_offset_inside'] + \
                                  np.random.rand(3) * (room_geometry - params['room_offset_inside'] * 2)
                #######################################################
                if params['rir_proc_dir'].split('_')[2] == 'same':
                    source_location[-1] = azz
                #######################################################
                src_dist = np.sqrt(np.sum(np.power(source_location - array_center_location, 2)))

                valid = 0
                line_new = source_location - array_center_location
                for j in range(len(source_location_list)):
                    line_old = source_location_list[j] - array_center_location
                    denominator = np.sqrt(np.sum(line_new ** 2)) * np.sqrt(np.sum(line_old ** 2)) + 1e-9
                    cos_angle = np.arccos(np.dot(line_new, line_old) / denominator) * 180 / math.pi # degree [0,180]
                    if cos_angle > params['angle_between_sources_min']:
                        valid += 1

                if (params['array_source_distance_min'] < src_dist < params['array_source_distance_max']) and \
                        (valid == len(source_location_list)):
                    break
                cnt += 1
                if cnt > max_tries_room:
                    assert 0, f"Speech source locating failed."

        # Add speech source"s"
        # reverb
        room1.add_source(position=source_location)
        source_location_list.append(source_location)


    # Compute RIRs
    # reverb
    room1.image_source_model()
    room1.compute_rir()

    dset, mixture_name = file.split('/')[3], file.split('/')[4]
    filename1 = f'RIR_{dset}_{mixture_name}_reverb'
    createDirectory(params['rir_proc_dir'])
    save_dir1 = os.path.join(params['rir_proc_dir'], filename1)
    saveFigure(params, room1, rt60, room_geometry, array_center_location, mic_angle, source_location_list, save_dir1)

    rirs1 = room1.rir
    # To make np.array & zero padding
    rir_len_ls1 = []
    for mic_idx in range(len(rirs1)):
        for src_idx in range(len(rirs1[mic_idx])):
            rir_len_ls1.append(len(rirs1[mic_idx][src_idx]))
    rir_len_max1 = max(rir_len_ls1)
    for mic_idx in range(len(rirs1)):
        for src_idx in range(len(rirs1[mic_idx])):
            rirs1[mic_idx][src_idx] = np.pad(rirs1[mic_idx][src_idx],
                                            (0, rir_len_max1 - len(rirs1[mic_idx][src_idx])),
                                            'constant')
    rirs1 = np.array(rirs1)
    rirs1 = np.swapaxes(rirs1, 0, 1) # [# src, # mic, length]

    try:
        np.save(save_dir1, rirs1)
    except Exception as e:
        print(str(e))


    angle_list = []
    for i in range(len(source_location_list)):
        line_org = source_location_list[i] - array_center_location
        line_xy = source_location_list[i] - array_center_location
        line_xy[2] = 0 # same z-axis value
        denominator = np.sqrt(np.sum(line_org ** 2)) * np.sqrt(np.sum(line_xy ** 2)) + 1e-9
        elevation_angle = np.arccos(np.dot(line_org, line_xy) / denominator) * 180 / math.pi  # degree [0,180(90)]
        # consider the sources below the mic array
        if line_org[2] < 0:
            elevation_angle = -elevation_angle # degree [-90,90]

        line_abs = np.array([1, 0, 0]) # refer to absolute x-axis
        denominator = np.sqrt(np.sum(line_abs ** 2)) * np.sqrt(np.sum(line_xy ** 2)) + 1e-9
        azimuth_angle = np.arccos(np.dot(line_abs, line_xy) / denominator) * 180 / math.pi  # degree [0,180]
        # consider the sources on negative y-axis of the mic array
        if line_org[1] < 0:
            azimuth_angle = 360 - azimuth_angle # degree [0,360]
        azimuth_angle = azimuth_angle - (mic_angle * 180 / math.pi) # degree [-mic_angle*180/pi,360-mic_angle*180/pi]
        if azimuth_angle < 0:
            azimuth_angle = 360 + azimuth_angle # degree [0,360]

        angle_list.append((azimuth_angle, elevation_angle))

    return save_dir1, array_center_location, mic_angle*180/math.pi, source_location_list, angle_list


def Resample(audio, orig_sr, target_sr):
    audio = torch.from_numpy(audio).type(torch.float32).cuda()
    transform = torchaudio.transforms.Resample(orig_sr, target_sr).cuda()
    return transform(audio).detach().cpu().numpy()
