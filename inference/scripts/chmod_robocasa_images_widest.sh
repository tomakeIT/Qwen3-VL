#!/usr/bin/env bash
# 将 Robocasa lerobot 图像目录下所有文件/子目录权限设为当前环境允许的最宽权限（等价 a+rwx / 777）。
# 大量文件时优先用单次递归 chmod，比 find 每文件 exec 一次快得多。
#
# 用法:
#   ./chmod_robocasa_images_widest.sh
#   ./chmod_robocasa_images_widest.sh /path/to/images

set -euo pipefail

TARGET="${1:-"~/LightwheelData/Robocasa_lerobot_6tasks/images"}"

# if [[ ! -d "$TARGET" ]]; then
#   echo "错误: 目录不存在: $TARGET" >&2
#   exit 1
# fi

echo "chmod -R a+rwx: $TARGET"
# 单次递归，内核/工具链对目录树遍历做了优化，通常比 find -exec chmod \; 快很多
sudo chmod -R a+rwx "$TARGET"
echo "完成。"
