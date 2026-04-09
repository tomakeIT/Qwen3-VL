import h5py
import json
import os

from datasets import EpisodeData, HDF5DatasetFileHandler

# def check_layout_style_ids(hdf5_file_path):
#     """
#     检查HDF5文件中的layout_id和style_id是否在0～10之间

#     Args:
#         hdf5_file_path: HDF5文件路径

#     Returns:
#         dict: 包含检查结果的信息
#     """
#     try:
#         with h5py.File(hdf5_file_path, 'r') as f:
#             # 读取env_args属性
#             print(f.attrs)
#             if 'data' in f.attrs:
#                 data = f.attrs['data']
#                 print(data)
#                 env_args_str = data.attrs['env_args']

#                 # 解析JSON字符串
#                 if isinstance(env_args_str, str):
#                     env_args = json.loads(env_args_str)
#                 else:
#                     env_args = env_args_str

#                 # 提取layout_id和style_id
#                 layout_id = env_args.get('layout_id')
#                 style_id = env_args.get('style_id')

#                 # 检查是否存在
#                 if layout_id is None or style_id is None:
#                     return {
#                         'file': hdf5_file_path,
#                         'layout_id': layout_id,
#                         'style_id': style_id,
#                         'layout_in_range': False,
#                         'style_in_range': False,
#                         'error': 'layout_id or style_id not found'
#                     }

#                 # 判断是否在0～10之间
#                 layout_in_range = 0 <= layout_id <= 10
#                 style_in_range = 0 <= style_id <= 10

#                 return {
#                     'file': hdf5_file_path,
#                     'layout_id': layout_id,
#                     'style_id': style_id,
#                     'layout_in_range': layout_in_range,
#                     'style_in_range': style_in_range,
#                     'both_in_range': layout_in_range and style_in_range
#                 }
#             else:
#                 return {
#                     'file': hdf5_file_path,
#                     'error': 'env_args attribute not found'
#                 }

#     except Exception as e:
#         return {
#             'file': hdf5_file_path,
#             'error': f'Error reading file: {str(e)}'
#         }


def check_layout_style_ids(hdf5_file_path):
    # Load dataset
    if not os.path.exists(hdf5_file_path):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(hdf5_file_path)
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    episode_names = list(dataset_file_handler.get_episode_names())
    episode_names.sort(key=lambda x: int(x.split("_")[-1]))

    env_args = json.loads(dataset_file_handler._hdf5_data_group.attrs["env_args"])

    # scene_name=f"{scene_name}-{env_args['layout_id']}-{env_args['style_id']}",
    # 提取layout_id和style_id
    layout_id = env_args['layout_id']
    style_id = env_args['style_id']

    # 检查是否存在
    if layout_id is None or style_id is None:
        return {
            'file': hdf5_file_path,
            'layout_id': layout_id,
            'style_id': style_id,
            'layout_in_range': False,
            'style_in_range': False,
            'error': 'layout_id or style_id not found'
        }

    # 判断是否在0～10之间
    layout_in_range = 0 <= layout_id <= 10
    style_in_range = 0 <= style_id <= 10

    return {
        'file': hdf5_file_path,
        'layout_id': layout_id,
        'style_id': style_id,
        'layout_in_range': layout_in_range,
        'style_in_range': style_in_range,
        'both_in_range': layout_in_range and style_in_range
    }


if __name__ == "__main__":
    # 使用示例
    result = check_layout_style_ids('/tmp/deliver/final/CloseFridge/CloseFridge_1759029789391107/dataset.hdf5')
    print(f"文件: {result['file']}")
    print(f"layout_id: {result.get('layout_id', 'N/A')}, 在0-10范围内: {result.get('layout_in_range', False)}")
    print(f"style_id: {result.get('style_id', 'N/A')}, 在0-10范围内: {result.get('style_in_range', False)}")
