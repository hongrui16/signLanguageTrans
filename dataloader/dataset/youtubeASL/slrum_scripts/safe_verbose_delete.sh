#!/bin/bash

# 设置你要删除的目录（请改成绝对路径）
TARGET_DIR="$1"

echo "🚀 Starting batch deletion under: $TARGET_DIR"

# 检查目录是否存在
if [ ! -d "$TARGET_DIR" ]; then
    echo "❌ Directory not found: $TARGET_DIR"
    exit 1
fi

# 初始文件统计
initial_count=$(find "$TARGET_DIR" -type f | wc -l)
echo "📂 Initial file count: $initial_count"

deleted_total=0
batch_size=1000

# 开始分批删除
while true; do
    files=$(find "$TARGET_DIR" -type f | head -n $batch_size)
    if [ -z "$files" ]; then
        echo "✅ All files deleted. Total deleted: $deleted_total"
        break
    fi

    echo "🧹 Deleting $batch_size files (current total: $deleted_total)..."

    # 逐个打印并删除
    while IFS= read -r file; do
        # echo "🗑️  Deleting: $file"
        rm -f "$file"
        deleted_total=$((deleted_total + 1))
    done <<< "$files"

    echo "✅ Batch complete. Total deleted: $deleted_total"
done
