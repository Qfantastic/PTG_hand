import numpy as np
import os
import yaml
import pandas as pd
import argparse

from scipy.spatial.transform import Rotation as Rot




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")

    args = parser.parse_args()
    return args

args = parse_args()
sequence_folder = './data/20220922/' + args.folder + '/'


with open(os.path.join(sequence_folder, "meta.yml"), "r") as f:
    meta = yaml.load(f, Loader=yaml.FullLoader)

# holo_serial = meta["holo_serials"][0]
#df_holo_tfs_full = pd.read_csv(os.path.join(sequence_folder,"holo_tfs_full.csv"),index_col=0,dtype='str')



def _parse_str_to_RT(ss, shape=(4, 4), dtype=np.float32):
    ss = filter(lambda item: item not in "[]\n", ss)
    return np.array(''.join(ss).split(), dtype=np.float32).reshape(shape)

def _get_holo_world_to_tag0():
    df_hololens2_color2world = pd.read_html(sequence_folder + 'hololens2_color2world.html')[0]
    df_hololens2_color2world.drop('Unnamed: 0', axis =1, inplace = True)

    df_hololens2_depth2world = pd.read_html(sequence_folder + 'hololens2_depth2world.html')[0]
    df_hololens2_depth2world.drop('Unnamed: 0', axis=1, inplace=True)

    df_hololens2_tag2color = pd.read_html(sequence_folder + 'hololens2_tag2color.html')[0]
    df_hololens2_tag2color.drop('Unnamed: 0', axis=1, inplace=True)

    world2tag0_avg = np.zeros((4, 4), dtype=np.float32)
    count = 0
    for column_name in df_hololens2_tag2color:
        if column_name in df_hololens2_color2world.columns:
            cam2world = eval(df_hololens2_color2world[str(column_name)][0])
            tag02cam = eval(df_hololens2_tag2color[str(column_name)][0])
            world2tag0 = np.linalg.inv(np.matmul(cam2world, tag02cam))
            world2tag0_avg += world2tag0
            count += 1

    return world2tag0_avg / count






    #
    #
    # tfs = df_holo_tfs_full.loc[
    #     ["tag2cam_color_{}".format(holo_serial), "cam2world_color_{}".format(holo_serial)]].dropna(axis='columns')
    # world2tag0_avg = np.zeros((4, 4), dtype=np.float32)
    # count = 0
    # for p in tfs:
    #     tag02cam = _parse_str_to_RT(tfs.loc["tag2cam_color_hololens", p])
    #     cam2world = _parse_str_to_RT(tfs.loc["cam2world_color_hololens", p])
    #     world2tag0 = np.linalg.inv(np.matmul(cam2world, tag02cam))
    #     world2tag0_avg += world2tag0
    #     count += 1
    # return world2tag0_avg / count



def _extract_RT_from_tag_detection(tag):
    position = tag["pose"]["pose"]["pose"]["position"]
    quaternion = tag["pose"]["pose"]["pose"]["orientation"]
    RT = np.eye(4)
    RT[:3, 3] = [position["x"], position["y"], position["z"]]
    RT[:3, :3] = Rot.from_quat(
        [quaternion["x"], quaternion["y"], quaternion["z"], quaternion["w"]]
    ).as_matrix()
    return RT.astype('float32')


def _extract_holoworld2tag(tag):

    world2tag0 = np.zeros((4, 4), dtype=np.float32)
    world2tag0[3][3] = 1
    rt = np.array(tag['holoworld2tag']['hololens2'])

    return rt.astype('float32')

def _extract_Trans_from_tag_detection_yml(tag_file):
    with open(tag_file, "r") as f:
        detections = list(yaml.load_all(f, Loader=yaml.SafeLoader))
    tag_0 = np.zeros([4, 4])
    tag_1 = np.zeros([4, 4])
    counter_0 = 0
    counter_1 = 0
    for detection in detections:
        try:
            tag_0 += _extract_RT_from_tag_detection(detection[0])
            counter_0 += 1
        except:
            pass
        try:
            tag_1 += _extract_RT_from_tag_detection(detection[1])
            counter_1 += 1
        except:
            pass
    tag_0 /= counter_0
    tag_1 /= counter_1
    return tag_0,tag_1


meta_file = open(sequence_folder + 'meta.yml').read()
meta_tag = yaml.safe_load(meta_file)


calibration_folder = './2022-04-17/'
tag0, tag1 = _extract_Trans_from_tag_detection_yml(os.path.join(calibration_folder, "apriltag_640x480.yml"))
# holo_world = _get_holo_world_to_tag0()
holo_world = _extract_holoworld2tag(meta_tag)
holo_world_to_cam0 = np.matmul(tag0,holo_world)
print(tag0)
print(holo_world)
print('holo_world_to_cam0',holo_world_to_cam0)

n_index = 0
print('the rotation:')
for i in range(3):
    for j in range(3):
        print(holo_world_to_cam0[i][j])
print('the translation:')
for i in range(3):
    print(holo_world_to_cam0[i][3])
    # save data/20220520_161851 as extrinsics/holoworld.yml
    # save data/20220529/20220529_143853 as extrinsics/holoworld1.yml
    # save data/20220529/20220529_144535 as extrinsics/holoworld2.yml
    # save data/20220529/20220529_144010 as extrinsics/holoworld3.yml
    # save data/20220529/20220529_144127 as extrinsics/holoworld4.yml
    # save data/20220529/20220529_144321 as extrinsics/holoworld5.yml
    # save data/20220529/20220529_144359 as extrinsics/holoworld6.yml
    # save data/20220529/20220529_144441 as extrinsics/holoworld7.yml
    # save data/20220529/20220529_144635 as extrinsics/holoworld8.yml



# recording_id = sequence_folder[-7:-1]

dict_file = {'rotation' : [float(holo_world_to_cam0[0][0]), float(holo_world_to_cam0[0][1]),
                            float(holo_world_to_cam0[0][2]), float(holo_world_to_cam0[1][0]),
                            float(holo_world_to_cam0[1][1]), float(holo_world_to_cam0[1][2]),
                            float(holo_world_to_cam0[2][0]), float(holo_world_to_cam0[2][1]),
                            float(holo_world_to_cam0[2][2])], 'serial' : 'holoworld',
             'translation' : [float(holo_world_to_cam0[0][3]), float(holo_world_to_cam0[1][3]), float(holo_world_to_cam0[2][3])]}






with open('2022-04-17/extrinsics/holoworld_'+ args.folder + '.yml', 'w') as outfile:
    yaml.dump(dict_file, outfile)

with open(r'2022-04-17/intrinsics/holoworld_640x480.yml') as file:
    documents = yaml.full_load(file)


with open('2022-04-17/intrinsics/holoworld_' + args.folder + '_640x480.yml', 'w') as outfile:
    yaml.dump(documents, outfile)



