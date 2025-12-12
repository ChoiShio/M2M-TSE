import os
import glob
import jams
import shutil
import argparse

import time
import datetime
from tqdm import tqdm
import numpy as np

from scipy import signal
import soundfile
# import librosa

from utils import createDirectory, deleteDirectory, getParams, generateRIRs, Resample

import torch
torch.set_num_threads(8)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["OMP_NUM_THREADS"] = '4'
os.environ["OPENBLAS_NUM_THREADS"] = '4'
os.environ["MKL_NUM_THREADS"] =  '4'
os.environ["VECLIB_MAXIMUM_THREADS"] = '4'
os.environ["NUMEXPR_NUM_THREADS"] = '4'

import random
seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



def option_0(jams_files, args):
    start = time.time()
    params = getParams(args)

    # Generate directory for option_0
    directory_MC = params['rir_proc_dir'].replace('jams_RIRs', 'FSDSoundScapes_MC')
    deleteDirectory(directory_MC)
    createDirectory(directory_MC)

    deleteDirectory(params['rir_proc_dir'])
    createDirectory(params['rir_proc_dir'])

    print('Copying audio & jams files...')
    shutil.copytree('FSD2018_TAU2019/FSDSoundScapes/FSDKaggle2018', os.path.join(directory_MC, 'FSDKaggle2018'))
    shutil.copytree('FSD2018_TAU2019/FSDSoundScapes/jams', os.path.join(directory_MC, 'jams'))

    # Generate zero noise (4-channel)
    zero_audio_MC = np.ravel(np.zeros(params['fs'] * 40)).reshape(params['fs'] * 10, 4)
    createDirectory(os.path.join(directory_MC, 'background'))
    zero_path_MC = os.path.join(directory_MC, 'background', 'zero.wav')
    soundfile.write(zero_path_MC, zero_audio_MC, params['fs'])

    # Generate RIRs and edit jams files
    print('Generating RIRs and editing jams files...')
    for file in tqdm(jams_files):
        jam_MC = jams.load(file)

        for annotation in jam_MC.annotations:
            foreground_list = []
            for idx, audio_dict in enumerate(annotation.data):
                if idx == 0:
                    audio_dict.value['source_file'] = zero_path_MC
                else:
                    foreground_list.append(os.path.join('FSD2018_TAU2019', audio_dict.value['source_file']))
                    audio_dict.value['source_file'] = \
                        audio_dict.value['source_file'].replace('FSDSoundScapes', directory_MC)

        save_dir, mic_center, mic_angle, coord_list, angle_list = generateRIRs(params, foreground_list, file)

        for annotation in jam_MC.annotations:
            for idx, audio_dict in enumerate(annotation.data):
                if idx != 0:
                    audio_dict.value['RIRs_dir'] = save_dir
                    audio_dict.value['RIRs_src_idx'] = idx - 1

                    audio_dict.value['x_coordinate'] = coord_list[idx-1][0]
                    audio_dict.value['y_coordinate'] = coord_list[idx-1][1]
                    audio_dict.value['z_coordinate'] = coord_list[idx-1][2]
                    audio_dict.value['azimuth_angle'] = angle_list[idx-1][0]
                    audio_dict.value['elevation_angle'] = angle_list[idx-1][1]

            annotation.sandbox.scaper['sr'] = params['fs']
            annotation.sandbox.scaper['n_channels'] = 4
            annotation.sandbox.scaper['mic_x_coordinate'] = mic_center[0]
            annotation.sandbox.scaper['mic_y_coordinate'] = mic_center[1]
            annotation.sandbox.scaper['mic_z_coordinate'] = mic_center[2]
            annotation.sandbox.scaper['mic_rotation_angle'] = mic_angle

        jam_save_path_MC = file.replace('FSD2018_TAU2019/FSDSoundScapes', directory_MC)
        jam_MC.save(jam_save_path_MC)

    end = time.time()
    total_time = str(datetime.timedelta(seconds=(end - start)))  # h:mm:ss.ms
    print(f"It takes {total_time} to process option_0.")


def option_1(jams_files, args):
    start = time.time()
    params = getParams(args)

    # Generate directory for option_1
    directory_MC = params['rir_proc_dir'].replace('jams_RIRs', 'FSDSoundScapes_MC') + '_noise'
    deleteDirectory(directory_MC)

    print('Copying audio & jams files...')
    shutil.copytree(params['rir_proc_dir'].replace('jams_RIRs', 'FSDSoundScapes_MC'), directory_MC)

    # Generate REVERB noise (4-channel)
    noise_4ch_list = sorted(glob.glob('REVERB_4ch/*.wav'))
    select_index = np.random.randint(0, len(noise_4ch_list), len(jams_files))

    deleteDirectory(os.path.join(directory_MC, 'background'))

    # Edit jams files
    print('Editing jams files...')
    jams_files_MC = sorted(glob.glob(os.path.join(directory_MC, 'jams/*/*/mixture.jams')),
                           key=lambda x: (x.split('/')[2],   # test, train, val
                                          x.split('/')[3]))  # 00000000 ~
    for ii, file in enumerate(tqdm(jams_files_MC)):
        jam = jams.load(file)

        for annotation in jam.annotations:
            for idx, audio_dict in enumerate(annotation.data):
                if idx == 0:
                    audio_dict.value['source_file'] = noise_4ch_list[select_index[ii]]
                else:
                    audio_dict.value['source_file'] = \
                        audio_dict.value['source_file'].replace(
                            params['rir_proc_dir'].replace('jams_RIRs', 'FSDSoundScapes_MC'), directory_MC)

        jam.save(file)

    end = time.time()
    total_time = str(datetime.timedelta(seconds=(end-start))) # h:mm:ss.ms
    print(f"It takes {total_time} to process option_1.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preparing dataset for TSE')

    ## related to data loading
    parser.add_argument('--option', type=int, default=0,
                        help=f'choose which data will be generated\n'
                             f'0: original + `zero` noise + RIR\n'
                             f'1: original + `REVERB` noise + RIR after option 0\n')
    parser.add_argument('--cfg', default='pyroom.cfg', help='Read pyroom.cfg for all the details')
    parser.add_argument('--cfg_str', type=str, default='pyroom')

    args = parser.parse_args()

    jams_files = sorted(glob.glob('FSD2018_TAU2019/FSDSoundScapes/jams/*/*/mixture.jams'),
                        key=lambda x: (x.split('/')[3],   # test, train, val
                                       x.split('/')[4]))  # 00000000 ~

    if args.option == 0:
        option_0(jams_files, args)
    elif args.option == 1:
        option_1(jams_files, args)
    else:
        print("Choose one option from 0 and 1 !!")
