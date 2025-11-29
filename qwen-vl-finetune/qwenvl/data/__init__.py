import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}
DEMO_DATA = {
    "annotation_path": "/home/erdao/Documents/Qwen3-VL/qwen-vl-finetune/demo/single_images.json",
    "data_path": "/home/erdao/Documents/Qwen3-VL/qwen-vl-finetune",
}
EXAMPLE_DATA = {
    "annotation_path": "/home/erdao/Documents/Qwen3-VL/qwen-vl-finetune/Example_data/dataset/train/L10K3TurnOnTheStoveAndPutTheMokaPotOnIt.json",
    "data_path": "/home/erdao/Documents/Qwen3-VL/qwen-vl-finetune",
}

# 4 tasks with negative samples
L90L6PutTheWhiteMugOnThePlate = {
        "annotation_path": "/home/lightwheel/erdao.liang/LightwheelData/dataset_1122/train/L90L6PutTheWhiteMugOnThePlate.json",
        "data_path": "/home/lightwheel/erdao.liang/LightwheelData"
    }

L10K8PutBothMokaPotsOnTheStove = {
    "annotation_path": "/home/lightwheel/erdao.liang/LightwheelData/dataset_1122/train/L10K8PutBothMokaPotsOnTheStove.json",
    "data_path": "/home/lightwheel/erdao.liang/LightwheelData"
}

L90K2StackTheMiddleBlackBowlOnTheBackBlackBowl = {
    "annotation_path": "/home/lightwheel/erdao.liang/LightwheelData/dataset_1122/train/L90K2StackTheMiddleBlackBowlOnTheBackBlackBowl.json",
    "data_path": "/home/lightwheel/erdao.liang/LightwheelData"
}

L90L3PickUpTheCreamCheeseAndPutItInTheTray = {
    "annotation_path": "/home/lightwheel/erdao.liang/LightwheelData/dataset_1122/train/L90L3PickUpTheCreamCheeseAndPutItInTheTray.json",
    "data_path": "/home/lightwheel/erdao.liang/LightwheelData"
}


data_dict = {
    # "demo": DEMO_DATA,
    # "example": EXAMPLE_DATA,
    # "data1": L10K3TurnOnTheStoveAndPutTheMokaPotOnIt,
    # "data2": L10K4PutTheBlackBowlInTheBottomDrawerOfTheCabinetAndCloseIt,
    # "data3": L10K6PutTheYellowAndWhiteMugInTheMicrowaveAndCloseIt,
    "put_white_mug_on_plate": L90L6PutTheWhiteMugOnThePlate,
    "put_both_moka_pots_on_stove": L10K8PutBothMokaPotsOnTheStove,
    "stack_middle_black_bowl_on_back_black_bowl": L90K2StackTheMiddleBlackBowlOnTheBackBlackBowl,
    "pick_up_cream_cheese_and_put_in_tray": L90L3PickUpTheCreamCheeseAndPutItInTheTray,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
