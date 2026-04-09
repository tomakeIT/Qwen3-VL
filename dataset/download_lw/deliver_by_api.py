import shutil
import requests
import json
import os
import tarfile
import tos
# from tos.utils import SizeAdapter
from urllib.parse import urlparse
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import glob
from datasets import EpisodeData, HDF5DatasetFileHandler

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # Suppress InsecureRequestWarning

def read_tasks_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            names = [line.strip() for line in file if line.strip()]
        return names
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return []
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return []


def check_layout_style_ids(hdf5_file_path):
    # Load dataset
    if not os.path.exists(hdf5_file_path):
        raise FileNotFoundError(f"The dataset file {hdf5_file_path} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(hdf5_file_path)
    episode_count = dataset_file_handler.get_num_episodes()
    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    env_args = json.loads(dataset_file_handler._hdf5_data_group.attrs["env_args"])

    layout_id = env_args['layout_id']
    style_id = env_args['style_id']

    # 检查是否存在
    if layout_id is None or style_id is None:
        print(f"Warning: layout_id or style_id not found in {hdf5_file_path}. Pass this data")
        return False

    # 判断是否在0～10之间
    layout_in_range = 1 <= layout_id <= 10
    style_in_range = 1 <= style_id <= 10
    if not (layout_in_range and style_in_range):
        print(f"Warning: layout_id:{layout_id}, style_id: {style_id} in {hdf5_file_path}. Pass this data")
    return layout_in_range and style_in_range


class DeliverByApi:
    def __init__(self, username, password, deliver_path, deliver_task_txt, status=1, download_type=0):
        self.api_url = "https://assetserver.lightwheel.net"
        self.username = username
        self.password = password

        self.deliver_path = deliver_path
        self.deliver_task_txt = deliver_task_txt
        self.deliver_t
        ask_list = read_tasks_from_txt(self.deliver_task_txt)
        if self.deliver_task_list:
            print(f"任务列表: {self.deliver_task_list}")
        self.status = status
        self.download_type = download_type

        # 初始化tos客户端
        self.client = self.getTosClient()
        self.bytedance_bucket = "vla-data-nimbus"

    def login(self):
        url = f"{self.api_url}/api/authenticate/v1/user/login"
        data = {
            "username": self.username,
            "password": self.password,
        }
        response = requests.post(url, json=data)
        return response.json()["token"]

    def get_project_uuid(self, project_name, token):
        url = f"{self.api_url}/api/asset/v1/project/get"
        headers = {
            "Authorization": token,
            "UserName": self.username,
            "Content-Type": "application/json",
        }
        data = {
            "page": 1,
            "page_size": 100,
            "search_request": {
                "name": project_name,
            },
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()["data"][0]["uuid"]

    def get_teleoperations(self, project_uuid, token):
        url = f"{self.api_url}/api/asset/v1/teleoperation/get"
        headers = {
            "Authorization": token,
            "UserName": self.username,
            "Content-Type": "application/json",
        }
        # search_request status: 0: 待处理, 1: 已通过, 2: 已打回, 6: 已通过-负样本
        page = 1
        page_size = 300
        result_data = []
        pbar = tqdm(desc="获取teleoperations", unit="页")
        try:
            while True:
                data = {
                    "page": page,
                    "page_size": page_size,
                    "project_uuid": project_uuid,
                    "search_request": {
                        "status": self.status,
                        "created_at_start": "2021-10-29T00:00:00+08:00",
                        "created_at_end": "2026-10-29T23:59:59+08:00",
                    },
                }
                response = requests.post(url, headers=headers, json=data)
                res = response.json()["data"]
                result_data.extend(res)
                pbar.update(1)
                pbar.set_postfix({"已获取": len(result_data)})
                if len(res) < page_size:
                    break
                page += 1
        finally:
            pbar.close()
        return result_data

    def get_hdf5_url(self, teleoperation_uuid, project_uuid, token):
        url = f"{self.api_url}/api/asset/v1/teleoperation/download"
        headers = {
            "Authorization": token,
            "UserName": self.username,
            "Content-Type": "application/json",
        }
        # download_type: 0: 全部, 1: hdf5 工程文件, 2: 视频文件, 3: 参考图片, 4: 问题图片, 5: i2i joint target video, 6: i2i action video, 7: 遥操作重刷hdf5工程文件, 8: 遥操交付视频
        data = {
            "version_uuids": [teleoperation_uuid],
            "project_uuid": project_uuid,
            "download_type": self.download_type,
        }
        response = requests.post(url, headers=headers, json=data)
        res = response.json()
        if len(res["downloadInfos"]) == 0:
            return None
        return res["downloadInfos"][0]["files"][0]["url"]

    def _process_single_teleoperation(self, teleoperation, project_uuid, token, pbar, lock):
        """处理单个teleoperation（用于并行处理）"""
        uuid = teleoperation["versionUuid"]
        
        try:
            # 用于判断 i2i 是否通过，可能会有通过的和不通过的分开的需求
            i2i_joint_target_passed = teleoperation["i2iJointTargetStatus"] == "passed"
            i2i_action_passed = teleoperation["i2iActionStatus"] == "passed"

            # 可以基于metadata的一些信息进行过滤，如果过滤掉，则跳过
            # metadata = teleoperation["metadata"]
            # metadata_dict = json.loads(metadata)
            # if metadata_dict.get("is_deleted", False):
            #     return False, "已删除"
            # params = metadata_dict.get("params", False)
            # if params and metadata_dict.get("task", False) not in self.deliver_task_list:
            #     return False, "不在任务列表中"

            # 视频下载链接
            video_url = teleoperation["video"][0]["url"]
            video_name = teleoperation["video"][0]["name"]
            # hdf5 下载链接
            hdf5_url = self.get_hdf5_url(uuid, project_uuid, token)
            if hdf5_url is None:
                with lock:
                    pbar.set_postfix({"状态": f"获取hdf5链接失败: {uuid[:8]}"})
                return False, f"获取hdf5链接失败: {uuid[:8]}"
            
            hdf5_name = hdf5_url.split("/")[5].split("?")[0]

            # 按taskname过滤
            # task_name = hdf5_name.split("_")[0]
            # if task_name not in self.deliver_task_list:
            #     return False, "任务名不在列表中"

            name = hdf5_name.split(".")[0]
            with lock:
                pbar.set_postfix({"当前": name[:30]})

            # 下载文件和视频
            processed_file_path = self.process_files(hdf5_url, video_url, video_name, name)
            if processed_file_path is None:
                return False, f"处理失败: {name[:30]}"
            else:
                self.save_metadata(processed_file_path, teleoperation)
                return True, f"处理成功: {name[:30]}"
        except Exception as e:
            return False, f"异常: {str(e)}"

    def deliver(self, teleoperations, project_uuid, token, max_workers=4):
        """并行处理teleoperations"""
        success_count = 0
        fail_count = 0
        lock = threading.Lock()
        
        with tqdm(total=len(teleoperations), desc="处理teleoperations", unit="项") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_teleop = {
                    executor.submit(self._process_single_teleoperation, teleop, project_uuid, token, pbar, lock): teleop
                    for teleop in teleoperations
                }
                
                # 处理完成的任务
                for future in as_completed(future_to_teleop):
                    success, message = future.result()
                    with lock:
                        if success:
                            success_count += 1
                        else:
                            fail_count += 1
                        pbar.set_postfix({"成功": success_count, "失败": fail_count, "状态": message[:40]})
                        pbar.update(1)
        
        print(f"\n处理完成: 成功 {success_count} 项, 失败 {fail_count} 项")

    # def upload_files(self, file_path, object_key):
    #     multi_result = self.client.create_multipart_upload(
    #         self.bytedance_bucket,
    #         object_key,
    #         acl=tos.ACLType.ACL_Bucket_Owner_Full_Control,
    #         storage_class=tos.StorageClassType.Storage_Class_Standard,
    #     )

    #     upload_id = multi_result.upload_id
    #     parts = []
    #     # 上传分片数据
    #     with open(file_path, "rb") as f:
    #         part_number = 1
    #         offset = 0
    #         total_size = os.path.getsize(f.name)
    #         part_size = 20 * 1024 * 1024
    #         while offset < total_size:
    #             num_to_upload = min(part_size, total_size - offset)
    #             out = self.client.upload_part(
    #                 self.bytedance_bucket,
    #                 object_key,
    #                 upload_id,
    #                 part_number,
    #                 content=SizeAdapter(f, num_to_upload, init_offset=offset),
    #             )
    #             parts.append(out)
    #             offset += num_to_upload
    #             part_number += 1
    #         self.client.complete_multipart_upload(self.bytedance_bucket, object_key, upload_id, parts)

    def process_files(self, hdf5_url, video_url, video_name, name):
        parsed_url = urlparse(hdf5_url)
        hdf5_filename = os.path.basename(parsed_url.path)
        
        # 下载并解压tar文件
        taskname = name.split("_")[0]
        extracted_dir = f"{self.deliver_path}/{taskname}/{name}"
        os.makedirs(extracted_dir, exist_ok=True)
        
        try:
            with requests.get(hdf5_url, stream=True, verify=False) as r:
                r.raise_for_status()
                mode = "r|gz" if hdf5_filename.endswith(".tar.gz") else "r|"
                with tarfile.open(fileobj=r.raw, mode=mode) as tar:
                    tar.extractall(extracted_dir)
        except Exception as e:
            print(f"解压hdf5失败：{str(e)}")
            import traceback
            traceback.print_exc()
            if os.path.exists(extracted_dir):
                shutil.rmtree(extracted_dir)
            return None

        # 检查解压后的目录结构：如果文件被同名子目录包裹，将内容上移一层（！！必要，有个别下载下来就是目录结构不同）
        possible_subdir = os.path.join(extracted_dir, name)
        if os.path.exists(possible_subdir) and os.path.isdir(possible_subdir):
            # tar文件包含同名子目录，将子目录内容上移一层
            for item in os.listdir(possible_subdir):
                src = os.path.join(possible_subdir, item)
                dst = os.path.join(extracted_dir, item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                if os.path.isdir(src):
                    shutil.move(src, dst)
                else:
                    shutil.move(src, dst)
            # 删除空的子目录
            os.rmdir(possible_subdir)

        hdf5_files = glob.glob(os.path.join(extracted_dir, "*.hdf5"))
        if not hdf5_files or len(hdf5_files) != 1:
            print(f"Warning: not find hdf5 file in {extracted_dir}")
            shutil.rmtree(extracted_dir)
            return None
        if not check_layout_style_ids(hdf5_files[0]):
            print(f"Warning: layout_id or style_id not in range in {hdf5_files[0]}")
            shutil.rmtree(extracted_dir)
            return None

        # 下载视频
        video_path = os.path.join(extracted_dir, video_name)
        try:
            with requests.get(video_url, stream=True, verify=False) as r:
                r.raise_for_status()
                with open(video_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            print(f"视频下载失败：{str(e)}")
            shutil.rmtree(extracted_dir)
            return None

        return extracted_dir

    def save_metadata(self, extracted_dir, teleoperation):
        metadata = {}
        if "issueDescription" in teleoperation:
            metadata["issueDescription"] = teleoperation["issueDescription"]
        
        metadata_path = os.path.join(extracted_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def extract_tar(self, tar_path, extract_path):
        # 根据文件扩展名选择正确的模式
        mode = "r:gz" if tar_path.endswith(".tar.gz") else "r:"
        with tarfile.open(tar_path, mode) as tar:
            tar.extractall(extract_path)

    def getTosClient(self):
        ak = os.environ.get("TOS_AK")
        sk = os.environ.get("TOS_SK")
        if not ak or not sk:
            raise ValueError("Missing TOS_AK or TOS_SK environment variables for authentication.")
        endpoint = "tos-cn-beijing.volces.com"
        region = "cn-beijing"

        # 初始化OSS客户端
        client = tos.TosClientV2(
            ak,
            sk,
            endpoint,
            region,
            # 通过security_token可选参数设置STS
            security_token=None,
            # 通过connection_time可选参数设置连接超时，单位：秒
            connection_time=10,
            # 通过socket_timeout可选参数设置Socket读写超时，单位：秒
            socket_timeout=30,
        )

        return client


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通过API交付机器人操作数据")
    parser.add_argument("--username", type=str, default="erdao.liang", help="用户名")
    parser.add_argument("--password", type=str, default="#Led070707", help="密码")
    parser.add_argument("--deliver-path", type=str, default="/home/erdao/Documents/LightwheelData/1W_Libero_X7s", help="交付路径")
    parser.add_argument("--deliver-task-txt", type=str, default="/home/erdao/Documents/LightwheelData/download_data/tasks.txt", help="任务列表文件路径")
    parser.add_argument("--project-name", type=str, default="1W_Libero_X7s", help="项目名称")
    parser.add_argument("--project-uuid", type=str, default=None, help="项目UUID（如果提供则跳过获取步骤）")
    parser.add_argument("--teleoperations-json", type=str, default=None, help="从JSON文件读取teleoperations（可选）")
    parser.add_argument("--max-workers", type=int, default=4, help="并行处理的线程数（默认4）")
    parser.add_argument("--status", type=int, default=1, help="search_request status: 0: 待处理, 1: 已通过, 2: 已打回, 6: 已通过-负样本")
    parser.add_argument("--download-type", type=int, default=0, help="download_type: 0: 全部, 1: hdf5 工程文件, 2: 视频文件, 3: 参考图片, 4: 问题图片, 5: i2i joint target video, 6: i2i action video, 7: 遥操作重刷hdf5工程文件, 8: 遥操交付视频")
    
    args = parser.parse_args()
    
    # 初始化
    deliver_by_api = DeliverByApi(
        username=args.username,
        password=args.password,
        deliver_path=args.deliver_path,
        deliver_task_txt=args.deliver_task_txt,
        status=args.status,
        download_type=args.download_type
    )
    
    # 获取认证的token
    token = deliver_by_api.login()
    print("登录成功")
    
    # 获取项目uuid
    if args.project_uuid:
        project_uuid = args.project_uuid
    else:
        project_uuid = deliver_by_api.get_project_uuid(args.project_name, token)
    print(f"project_uuid: {project_uuid}")

    # 获取或读取teleoperations
    if args.teleoperations_json:
        with open(args.teleoperations_json, "r", encoding="utf-8") as f:
            teleoperations = json.load(f)
        print(f"从JSON读取了 {len(teleoperations)} 条teleoperations")
    else:
        teleoperations = deliver_by_api.get_teleoperations(project_uuid, token)
        print(f"获取了 {len(teleoperations)} 条teleoperations")
        with open("teleoperations.json", "w", encoding="utf-8") as f:
            json.dump(teleoperations, f, indent=2, ensure_ascii=False)
    
    # 开始处理
    deliver_by_api.deliver(teleoperations, project_uuid, token, max_workers=args.max_workers)
