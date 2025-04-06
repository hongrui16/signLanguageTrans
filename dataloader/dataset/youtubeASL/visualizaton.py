import os
import re
import glob
from collections import defaultdict

# 图片目录
image_dir = "./output2"
output_html = "compare_by_scene.html"

# 匹配格式
pattern = re.compile(r'(filtered|uniform)_(.*?)_(C\d{8})_fid_(\d+)\.jpg')

# 获取图片路径
filtered_images = sorted(glob.glob(os.path.join(image_dir, "filtered*.jpg")))
uniform_images = sorted(glob.glob(os.path.join(image_dir, "uniform*.jpg")))

filtered_dict = defaultdict(dict)
uniform_dict = defaultdict(dict)
all_scene_ids = set()

# 解析 filtered
for path in filtered_images:
    filename = os.path.basename(path)
    match = pattern.match(filename)
    if match:
        _, _, scene_id, fid = match.groups()
        filtered_dict[scene_id][int(fid)] = path
        all_scene_ids.add(scene_id)

# 解析 uniform
for path in uniform_images:
    filename = os.path.basename(path)
    match = pattern.match(filename)
    if match:
        _, _, scene_id, fid = match.groups()
        uniform_dict[scene_id][int(fid)] = path
        all_scene_ids.add(scene_id)

# 按场景排序
all_scene_ids = sorted(all_scene_ids)

# HTML header
html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Filtered vs Uniform</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 20px; }
    h1 { text-align: center; }
    .scene-title {
      margin-top: 40px;
      margin-bottom: 10px;
      font-size: 18px;
      font-weight: bold;
      border-bottom: 1px solid #ccc;
      padding-bottom: 5px;
    }
    .row {
      display: flex;
      margin-bottom: 20px;
    }
    .column {
      flex: 1;
      text-align: center;
    }
    img {
      max-width: 90%;
      height: auto;
      border: 1px solid #ddd;
      padding: 4px;
      background: #f9f9f9;
    }
    .empty {
      height: 200px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #999;
      border: 1px dashed #ccc;
    }
    p {
      font-size: 12px;
      margin: 5px 0;
      word-break: break-all;
    }
  </style>
</head>
<body>
  <h1>Filtered vs Uniform Image Comparison</h1>
"""

# 每个场景
for scene_id in all_scene_ids:
    html += f'<div class="scene-title">Scene: {scene_id}</div>\n'

    fids = sorted(set(filtered_dict[scene_id].keys()).union(uniform_dict[scene_id].keys()))
    for fid in fids:
        f_img = filtered_dict[scene_id].get(fid)
        u_img = uniform_dict[scene_id].get(fid)

        html += '<div class="row">\n'

        # Filtered
        html += '<div class="column">\n'
        if f_img:
            rel_path = os.path.relpath(f_img, '.')  # 相对路径
            html += f'<p>{os.path.basename(f_img)}</p>\n'
            html += f'<img src="{rel_path}" alt="{rel_path}">\n'
        else:
            html += '<div class="empty">Empty</div>\n'
        html += '</div>\n'

        # Uniform
        html += '<div class="column">\n'
        if u_img:
            rel_path = os.path.relpath(u_img, '.')  # 相对路径
            html += f'<p>{os.path.basename(u_img)}</p>\n'
            html += f'<img src="{rel_path}" alt="{rel_path}">\n'
        else:
            html += '<div class="empty">Empty</div>\n'
        html += '</div>\n'

        html += '</div>\n'  # end of row

html += """
</body>
</html>
"""

# 写入文件
with open(output_html, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ HTML saved to: {output_html}. Open it in a browser to view results.")
