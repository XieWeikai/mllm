#!/bin/bash

# 检查是否提供了可执行文件路径
if [ $# -eq 0 ]; then
    echo "Usage: $0 <bin_file> [arg...]"
    exit 1
fi

# 获取可执行文件路径
BIN_FILE=$1
shift  # 移除第一个参数（bin_file），剩下的参数传递给可执行文件
ARGS=$@

# 检查文件是否存在
if [ ! -f "$BIN_FILE" ]; then
    echo "Error: File '$BIN_FILE' not found!"
    exit 1
fi

# 设备上的目标目录
DEVICE_DIR="/data/local/tmp/mllm/bin"

# 推送文件到设备
echo "Pushing $BIN_FILE to device..."
adb push "$BIN_FILE" "$DEVICE_DIR/"

# 检查 adb push 是否成功
if [ $? -ne 0 ]; then
    echo "Error: Failed to push $BIN_FILE to device!"
    exit 1
fi

# 获取可执行文件的文件名（去掉路径）
BIN_NAME=$(basename "$BIN_FILE")

# 在设备上执行
echo "Running $BIN_NAME on device with args: $ARGS..."
adb shell "cd $DEVICE_DIR && chmod +x $BIN_NAME && ./$BIN_NAME $ARGS"

# 检查 adb shell 是否成功
if [ $? -ne 0 ]; then
    echo "Error: Failed to run $BIN_NAME on device!"
    exit 1
fi
