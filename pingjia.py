from skimage.metrics import structural_similarity as compare_ssim, peak_signal_noise_ratio as calculate_psnr
import lpips
import torch
import cv2
import os

# 初始化 LPIPS 模型
lpips_model = lpips.LPIPS(net="alex")  # 可选择 "alex" 或 "vgg"
lpips_model.eval()  # 设置为评估模式

# 计算 SSIM
def calculate_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim, _ = compare_ssim(gray1, gray2, full=True)
    return ssim


# 计算 LPIPS
def calculate_lpips(img1, img2, lpips_model):
    img1 = img1[:, :, ::-1].copy()  # BGR 转 RGB，并创建副本
    img2 = img2[:, :, ::-1].copy()  # BGR 转 RGB，并创建副本
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    lpips_score = lpips_model(img1_tensor, img2_tensor)
    return lpips_score.item()


# 计算 PSNR
def calculate_psnr_value(img1, img2):
    psnr_value = calculate_psnr(img1, img2)
    return psnr_value


# 批量处理图片并计算均值
def batch_process(image_folder, groundtruth_folder):
    results = []
    total_psnr, total_ssim, total_lpips = 0, 0, 0
    count = 0

    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        groundtruth_path = os.path.join(groundtruth_folder, img_name)

        if os.path.exists(img_path) and os.path.exists(groundtruth_path):
            print(f"正在处理图片: {img_name}")
            img = cv2.imread(img_path)
            groundtruth = cv2.imread(groundtruth_path)

            # 计算 SSIM
            ssim_score = calculate_ssim(img, groundtruth)

            # 计算 LPIPS
            lpips_score = calculate_lpips(img, groundtruth, lpips_model)

            # 计算 PSNR
            psnr_score = calculate_psnr_value(img, groundtruth)

            # 累加评分
            total_psnr += psnr_score
            total_ssim += ssim_score
            total_lpips += lpips_score
            count += 1

            results.append({
                "image_name": img_name,
                "psnr": psnr_score,
                "ssim": ssim_score,
                "lpips": lpips_score
            })

    # 计算均值
    mean_psnr = total_psnr / count if count > 0 else 0
    mean_ssim = total_ssim / count if count > 0 else 0
    mean_lpips = total_lpips / count if count > 0 else 0

    return results, mean_psnr, mean_ssim, mean_lpips


# 使用示例
if __name__ == "__main__":
    image_folder = "E:/CDW2014-result/TRPCA/"  # 生成图片的文件夹路径
    groundtruth_folder = "E:/CDW2014-result/gt/"  # groundtruth 图片文件夹路径

    metrics, mean_psnr, mean_ssim, mean_lpips = batch_process(image_folder, groundtruth_folder)
    for result in metrics:
        print(
            f"图片: {result['image_name']}, PSNR: {result['psnr']:.4f}, SSIM: {result['ssim']:.4f}, LPIPS: {result['lpips']:.4f}")

    print("\n=== 平均值 ===")
    print(f"平均 PSNR: {mean_psnr:.4f}")
    print(f"平均 SSIM: {mean_ssim:.4f}")
    print(f"平均 LPIPS: {mean_lpips:.4f}")
# 使用示例
