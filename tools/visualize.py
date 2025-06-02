import re
import matplotlib.pyplot as plt


def visualize_loss(log_file, save_path=None):
    with open(log_file, "r") as f:
        log_text = f.read()

        # 使用正则提取 Epoch 和 Loss
        pattern = re.compile(r"train Epoch (\d+), Loss: ([\d.]+)")
        matches = pattern.findall(log_text)

        # 转换为数值
        epochs = [int(epoch) for epoch, _ in matches]
        losses = [float(loss) for _, loss in matches]

        # 绘图
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, losses, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        if save_path:
            plt.savefig(save_path)
            print(f"Loss curve saved to {save_path}")
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":
    # 示例用法
    log_file = "/home/rhong5/research_pro/hand_modeling_pro/signLanguageTrans/Zlog/20250401-111049_ID-3383124/info_2.log"  # 替换为你的日志文件路径
    save_path = "/home/rhong5/research_pro/hand_modeling_pro/signLanguageTrans/Zlog/20250401-111049_ID-3383124/composed_loss.jpg"  # 替换为你想保存的路径
    # visualize_loss(log_file, save_path)


    target_classname = [
            "0002_good_luck", "0003_fake_gun", "0004_star_trek", 
            "0005_star_trek_extended_thumb", "0006_thumbup_relaxed", "0007_thumbup_normal", "0008_thumbup_rigid", 
            "0009_thumbtucknormal", "0010_thumbtuckrigid", "0011_aokay", "0012_aokay_upright", "0013_surfer", 
            "0014_rocker", "0014_rocker_frontside", "0015_rocker_backside", "0016_fist", "0017_fist_rigid", 
            "0019_alligator_closed", "0023_one_count", "0024_two_count", "0025_three_count", "0026_four_count", 
            "0027_five_count", "0029_indextip", "0030_middletip", "0031_ringtip", "0032_pinkytip", "0035_palmdown", 
            "0037_fingerspreadrelaxed", "0038_fingerspreadnormal", "0039_fingerspreadrigid", "0040_capisce", 
            "0041_claws", "0043_peacock", "0044_cup", "0045_shakespearesyorick", "0051_dinosaur", "0058_middlefinger", 
            "0100_neutral_relaxed", "0101_neutral_rigid", "0102_good_luck", "0103_fake_gun", "0104_star_trek", 
            "0105_star_trek_extended_thumb", "0106_thumbup_relaxed", "0107_thumbup_normal", "0108_thumbup_rigid", 
            "0109_thumbtucknormal", "0110_thumbtuckrigid", "0111_aokay", "0112_aokay_upright", "0113_surfer", 
            "0114_rocker", "0115_rocker_backside", "0116_fist", "0117_fist_rigid", "0119_alligator_closed", 
            "0123_one_count", "0124_two_count", "0125_three_count", "0126_four_count", "0127_five_count", 
            "0129_indextip", "0130_middletip", "0131_ringtip", "0132_pinkytip", "0135_palmdown", "0137_fingerspreadrelaxed", 
            "0138_fingerspreadnormal", "0139_fingerspreadrigid", "0140_capisce", "0141_claws", "0143_peacock", 
            "0144_cup", "0145_shakespearesyorick", "0151_dinosaur", "0158_middlefinger",
        ]   
    print(len(target_classname))
    