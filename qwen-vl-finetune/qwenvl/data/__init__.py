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
        "annotation_path": "/home/erdao.liang/LightwheelData/dataset_1122/train/L90L6PutTheWhiteMugOnThePlate.json",
        "data_path": "/home/erdao.liang/LightwheelData"
    }

L10K8PutBothMokaPotsOnTheStove = {
    "annotation_path": "/home/erdao.liang/LightwheelData/dataset_1122/train/L10K8PutBothMokaPotsOnTheStove.json",
    "data_path": "/home/erdao.liang/LightwheelData"
}

L90K2StackTheMiddleBlackBowlOnTheBackBlackBowl = {
    "annotation_path": "/home/erdao.liang/LightwheelData/dataset_1122/train/L90K2StackTheMiddleBlackBowlOnTheBackBlackBowl.json",
    "data_path": "/home/erdao.liang/LightwheelData"
}

L90L3PickUpTheCreamCheeseAndPutItInTheTray = {
    "annotation_path": "/home/erdao.liang/LightwheelData/dataset_1122/train/L90L3PickUpTheCreamCheeseAndPutItInTheTray.json",
    "data_path": "/home/erdao.liang/LightwheelData"
}

ExampleDataPickUpTheCube = {
    "annotation_path": "/home/erdao.liang/LightwheelData/dataset_pickup_cube/train/GrabTheBlockAndLiftItUp.json",
    "data_path": "/home/erdao.liang/LightwheelData",
}

LSPickUpBlackBowlBetweenPlateAndRamekinAndPlaceItOnPlate = {
    "annotation_path": "/home/erdao.liang/LightwheelData/lerobot_old/train/LSPickUpBlackBowlBetweenPlateAndRamekinAndPlaceItOnPlate.json",
    "data_path": "/home/erdao.liang/LightwheelData/lerobot_old",
}

PickCoffeeMug = {
    "annotation_path": "/home/erdao.liang/LightwheelData/lerobot_old/train/PickCoffeeMug.json",
    "data_path": "/home/erdao.liang/LightwheelData/lerobot_old",
}

ArrangeVegetables = {
    "annotation_path": "/home/erdao.liang/LightwheelData/slowdata/data_1W_Robocasa_X7s_new/train/ArrangeVegetables.json",
    "data_path": "/home/erdao.liang/LightwheelData/slowdata/",
}

BreadAndCheese = {
    "annotation_path": "/home/erdao.liang/LightwheelData/slowdata/data_1W_Robocasa_X7s/train/BreadAndCheese.json",
    "data_path": "/home/erdao.liang/LightwheelData/slowdata/",
}

BreadSetupSlicing = {
    "annotation_path": "/home/erdao.liang/LightwheelData/slowdata/data_1W_Robocasa_X7s/train/BreadSetupSlicing.json",
    "data_path": "/home/erdao.liang/LightwheelData/slowdata/",
}

CheesyBread = {
    "annotation_path": "/home/erdao.liang/LightwheelData/slowdata/data_1W_Robocasa_X7s/train/CheesyBread.json",
    "data_path": "/home/erdao.liang/LightwheelData/slowdata/",
}

CloseDishwasher = {
    "annotation_path": "/home/erdao.liang/LightwheelData/slowdata/data_1W_Robocasa_X7s/train/CloseDishwasher.json",
    "data_path": "/home/erdao.liang/LightwheelData/slowdata/",
}

CloseDrawer = {
    "annotation_path": "/home/erdao.liang/LightwheelData/slowdata/data_1W_Robocasa_X7s/train/CloseDrawer.json",
    "data_path": "/home/erdao.liang/LightwheelData/slowdata/",
}

CloseFridge = {
    "annotation_path": "/home/erdao.liang/LightwheelData/slowdata/data_1W_Robocasa_X7s/train/CloseFridge.json",
    "data_path": "/home/erdao.liang/LightwheelData/slowdata/",
}

CloseMicrowave = {
    "annotation_path": "/home/erdao.liang/LightwheelData/slowdata/data_1W_Robocasa_X7s/train/CloseMicrowave.json",
    "data_path": "/home/erdao.liang/LightwheelData/slowdata/",
}

CoffeeServeMug = {
    "annotation_path": "/home/erdao.liang/LightwheelData/slowdata/data_1W_Robocasa_X7s/train/CoffeeServeMug.json",
    "data_path": "/home/erdao.liang/LightwheelData/slowdata/",
}

CoffeeSetupMug = {
    "annotation_path": "/home/erdao.liang/LightwheelData/slowdata/data_1W_Robocasa_X7s/train/CoffeeSetupMug.json",
    "data_path": "/home/erdao.liang/LightwheelData/slowdata/",
}

OpenDishwasher = {
    "annotation_path": "/home/erdao.liang/LightwheelData/slowdata/data_1W_Robocasa_X7s/train/OpenDishwasher.json",
    "data_path": "/home/erdao.liang/LightwheelData/slowdata/",
}

OpenDrawer = {
    "annotation_path": "/home/erdao.liang/LightwheelData/slowdata/data_1W_Robocasa_X7s/train/OpenDrawer.json",
    "data_path": "/home/erdao.liang/LightwheelData/slowdata/",
}

OpenFridge = {
    "annotation_path": "/home/erdao.liang/LightwheelData/slowdata/data_1W_Robocasa_X7s/train/OpenFridge.json",
    "data_path": "/home/erdao.liang/LightwheelData/slowdata/",
}

OpenMicrowave = {
    "annotation_path": "/home/erdao.liang/LightwheelData/slowdata/data_1W_Robocasa_X7s/train/OpenMicrowave.json",
    "data_path": "/home/erdao.liang/LightwheelData/slowdata/",
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
    "pick_up_cube": ExampleDataPickUpTheCube,
    "pick_up_black_bowl_between_plate_and_ramekin_and_place_it_on_plate": LSPickUpBlackBowlBetweenPlateAndRamekinAndPlaceItOnPlate,
    "pick_up_coffee_mug": PickCoffeeMug,
    "robocasa_x7s_arrange_vegetables": ArrangeVegetables,
    "robocasa_x7s_bread_and_cheese": BreadAndCheese,
    "robocasa_x7s_bread_setup_slicing": BreadSetupSlicing,
    "robocasa_x7s_cheesy_bread": CheesyBread,
    "robocasa_x7s_close_dishwasher": CloseDishwasher,
    "robocasa_x7s_close_drawer": CloseDrawer,
    "robocasa_x7s_close_fridge": CloseFridge,
    "robocasa_x7s_close_microwave": CloseMicrowave,
    "robocasa_x7s_coffee_serve_mug": CoffeeServeMug,
    "robocasa_x7s_coffee_setup_mug": CoffeeSetupMug,
    "robocasa_x7s_open_dishwasher": OpenDishwasher,
    "robocasa_x7s_open_drawer": OpenDrawer,
    "robocasa_x7s_open_fridge": OpenFridge,
    "robocasa_x7s_open_microwave": OpenMicrowave,
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
