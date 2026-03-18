#!/usr/bin/env python3
"""
植物盖度计算脚本 - 最终优化版
功能：计算卫星图像中红色区域内的植物盖度、植物面积和校园总面积
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os


def load_image(image_path):
    """加载图像"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")
    return img


def detect_red_lines(img):
    """
    检测图像中的纯红色线条 #FF0000（校园边界）
    只检测接近纯红色的像素，忽略其他红色调
    """
    # 方法1: 使用RGB直接检测纯红色 #FF0000
    # BGR格式中 #FF0000 = (0, 0, 255)
    b, g, r = cv2.split(img)
    
    # 纯红色特征: R通道高，G和B通道低
    # 允许一定的容差
    red_mask = np.zeros_like(b)
    
    # R > 200, G < 50, B < 50 (纯红色，允许一定容差)
    red_mask = cv2.inRange(img, np.array([0, 0, 200]), np.array([50, 50, 255]))
    
    # 方法2: 使用HSV进行更精确的纯红色检测
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 纯红色的HSV: H接近0或180，S接近255，V接近255
    # 只检测非常接近纯红色的像素
    lower_pure_red1 = np.array([0, 200, 200])
    upper_pure_red1 = np.array([5, 255, 255])
    mask1 = cv2.inRange(hsv, lower_pure_red1, upper_pure_red1)
    
    lower_pure_red2 = np.array([175, 200, 200])
    upper_pure_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_pure_red2, upper_pure_red2)
    
    hsv_red_mask = cv2.bitwise_or(mask1, mask2)
    
    # 合并两种方法的结果
    red_mask = cv2.bitwise_or(red_mask, hsv_red_mask)
    
    # 形态学操作连接线条
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)
    
    return red_mask


def find_campus_polygon(red_mask, img_shape):
    """
    从红色线条中找到校园边界多边形
    使用形态学操作填充线条围成的区域
    """
    # 查找轮廓
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("未找到红色边界")
    
    # 创建掩码
    campus_mask = np.zeros(img_shape[:2], dtype=np.uint8)
    
    # 方法：膨胀红色线条形成封闭边界，然后填充内部
    # 计算需要的膨胀程度 - 基于图像大小
    height, width = img_shape[:2]
    
    # 使用较大的核进行膨胀，使红色线条连接成封闭边界
    kernel_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 膨胀操作
    dilated = cv2.dilate(red_mask, kernel, iterations=2)
    
    # 使用闭操作进一步连接
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 查找膨胀后的轮廓
    contours_dilated, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours_dilated:
        # 找到最大的轮廓
        largest_contour = max(contours_dilated, key=cv2.contourArea)
        
        # 计算轮廓的周长
        perimeter = cv2.arcLength(largest_contour, True)
        
        # 简化多边形
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 填充多边形
        cv2.fillPoly(campus_mask, [approx], 255)
        
        # 腐蚀回来，缩小到原始边界内部
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        campus_mask = cv2.erode(campus_mask, erode_kernel, iterations=2)
        
        return campus_mask, approx
    
    # 如果失败，使用凸包方法
    red_points = np.where(red_mask > 0)
    if len(red_points[0]) > 0:
        points = np.column_stack((red_points[1], red_points[0]))
        hull = cv2.convexHull(points)
        cv2.fillPoly(campus_mask, [hull], 255)
        return campus_mask, hull
    
    return campus_mask, None


def detect_vegetation(img, campus_mask):
    """
    检测绿色植被区域
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 绿色范围 - 覆盖草地和树木
    lower_green = np.array([30, 35, 35])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 只保留校园区域内的植被
    vegetation_mask = cv2.bitwise_and(green_mask, campus_mask)
    
    return vegetation_mask


def calibrate_scale(img, scale_meters=100):
    """
    从左上角比例尺标定
    返回 scale_meters 米对应的像素数
    scale_meters: 比例尺的实际长度（米），默认100米
    """
    height, width = img.shape[:2]
    
    # 提取左上角区域 - 比例尺通常在左上角
    scale_region = img[:int(height*0.1), :int(width*0.2)]
    
    # 保存比例尺区域用于调试
    cv2.imwrite("debug_scale_region.png", scale_region)
    
    # 使用更宽松的红色检测范围
    hsv = cv2.cvtColor(scale_region, cv2.COLOR_BGR2HSV)
    
    # 放宽红色的HSV范围
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([15, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # 膨胀操作连接相邻的红色区域
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)
    
    # 保存红色检测结果
    cv2.imwrite("debug_scale_red.png", red_mask)
    
    # 查找轮廓
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 找到最长的水平线（比例尺通常是水平线）
    # 只考虑细长的水平线段（宽度远大于高度）
    max_length = 0
    all_lengths = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 水平线：宽度远大于高度（至少3倍）
        if w > h * 3:
            all_lengths.append((w, x, y, w, h))
            if w > max_length:
                max_length = w
    
    # 打印检测到的水平线
    if all_lengths:
        print(f"  Found {len(all_lengths)} horizontal red lines")
        for w, x, y, width, height in sorted(all_lengths, reverse=True)[:5]:
            print(f"    - Length: {w}px at ({x},{y}), size: {width}x{height}")
    
    # 如果检测失败，使用默认值
    if max_length < 30:
        print("  Warning: Auto scale detection failed, using default value")
        max_length = 95
    else:
        print(f"  Detected scale bar: {max_length} pixels = {scale_meters} meters")
    
    return max_length, scale_meters


def calculate_coverage(vegetation_mask, campus_mask, scale_pixels, scale_meters):
    """
    计算盖度和面积
    scale_pixels: 比例尺的像素长度
    scale_meters: 比例尺的实际长度（米）
    """
    vegetation_pixels = np.sum(vegetation_mask > 0)
    campus_pixels = np.sum(campus_mask > 0)
    
    coverage_percent = (vegetation_pixels / campus_pixels) * 100 if campus_pixels > 0 else 0
    
    # 计算实际面积
    pixels_per_meter = scale_pixels / scale_meters
    area_per_pixel_sqm = (1 / pixels_per_meter) ** 2
    
    vegetation_area = vegetation_pixels * area_per_pixel_sqm
    campus_area = campus_pixels * area_per_pixel_sqm
    
    return {
        'vegetation_pixels': int(vegetation_pixels),
        'campus_pixels': int(campus_pixels),
        'coverage_percent': coverage_percent,
        'vegetation_area': vegetation_area,
        'campus_area': campus_area,
        'scale_pixels': scale_pixels,
        'scale_meters': scale_meters,
        'area_per_pixel': area_per_pixel_sqm
    }


def visualize_results(img, campus_mask, vegetation_mask, red_mask, results, output_path=None):
    """
    可视化结果 - 简洁版
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 创建图形
    fig = plt.figure(figsize=(14, 8))
    
    # 使用 GridSpec 创建布局
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.3], 
                          width_ratios=[1, 1],
                          hspace=0.1, wspace=0.1,
                          left=0.05, right=0.95, top=0.9, bottom=0.1)
    
    # 第一行：原始图像 + 植被检测
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(vegetation_mask, cmap='Greens')
    ax2.set_title('Vegetation Mask', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 第二行：结果统计
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    result_text = (
        f"{'='*70}\n"
        f"  VEGETATION COVERAGE ANALYSIS RESULTS\n"
        f"{'='*70}\n\n"
        f"  Vegetation Coverage:  {results['coverage_percent']:.2f}%\n\n"
        f"  Vegetation Area:      {results['vegetation_area']:,.2f} m²  ({results['vegetation_area']/10000:.2f} ha)\n"
        f"  Campus Total Area:    {results['campus_area']:,.2f} m²  ({results['campus_area']/10000:.2f} ha)\n\n"
        f"  Scale: {results['scale_pixels']:.1f} pixels = {results['scale_meters']} meters    |    "
        f"Resolution: {results['area_per_pixel']:.3f} m²/pixel\n"
        f"{'='*70}"
    )
    
    ax3.text(0.5, 0.5, result_text, transform=ax3.transAxes,
             fontsize=12, verticalalignment='center', horizontalalignment='center',
             family='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#f0f8ff', 
                      edgecolor='#4169e1', alpha=0.95, linewidth=2))
    
    fig.suptitle('Vegetation Coverage Analysis - XJTLU Campus', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if output_path:
        plt.savefig(output_path, dpi=150, facecolor='white', edgecolor='none')
        print(f"Result saved to: {output_path}")
    
    plt.show()


def main(image_path=None, scale_pixels=None, scale_meters=100):
    """主函数
    image_path: 图片路径
    scale_pixels: 比例尺的像素长度（可选，不指定则交互式输入）
    scale_meters: 比例尺的实际长度（米），默认100米
    """
    if image_path is None:
        args = sys.argv[1:]
        if len(args) > 0:
            image_path = args[0]
        else:
            image_path = "截屏2026-03-10 00.47.12.png"
        
        # 解析命令行参数
        # 格式: python script.py image_path [scale_pixels] [scale_meters]
        if len(args) >= 3:
            scale_pixels = int(args[1])
            scale_meters = int(args[2])
        elif len(args) == 2:
            scale_meters = int(args[1])
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    # 交互式输入比例尺
    if scale_pixels is None:
        print("\n" + "=" * 60)
        print("请输入比例尺信息")
        print("=" * 60)
        print(f"当前图片: {image_path}")
        print("\n请查看图片左上角的比例尺，然后输入以下信息：")
        
        try:
            scale_pixels = int(input("比例尺像素长度 (像素): "))
            scale_meters = int(input("比例尺实际长度 (米): "))
        except ValueError:
            print("输入无效，使用默认值: 223像素 = 100米")
            scale_pixels = 223
            scale_meters = 100
    
    output_name = os.path.splitext(os.path.basename(image_path))[0] + "_result.png"
    
    print("\n" + "=" * 60)
    print("Vegetation Coverage Analysis Tool")
    print("=" * 60)
    
    # 1. 加载图像
    print("\n[1/5] Loading image...")
    img = load_image(image_path)
    height, width = img.shape[:2]
    print(f"Image size: {width} x {height} pixels")
    
    # 2. 检测红色边界
    print("\n[2/5] Detecting red boundary lines...")
    red_mask = detect_red_lines(img)
    
    # 3. 提取校园区域
    print("\n[3/5] Extracting campus area...")
    campus_mask, campus_poly = find_campus_polygon(red_mask, img.shape)
    campus_pixels = np.sum(campus_mask > 0)
    print(f"Campus area: {campus_pixels:,} pixels")
    
    # 4. 标定比例尺
    print("\n[4/5] Calibrating scale...")
    if scale_pixels is None:
        scale_pixels, scale_meters = calibrate_scale(img, scale_meters)
    else:
        print(f"  Using manual scale: {scale_pixels} pixels = {scale_meters} meters")
    print(f"Scale: {scale_pixels} pixels = {scale_meters} meters")
    
    # 5. 检测植被
    print("\n[5/5] Detecting vegetation...")
    vegetation_mask = detect_vegetation(img, campus_mask)
    vegetation_pixels = np.sum(vegetation_mask > 0)
    print(f"Vegetation pixels: {vegetation_pixels:,}")
    
    # 计算结果
    results = calculate_coverage(vegetation_mask, campus_mask, scale_pixels, scale_meters)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\n🌿 Vegetation Coverage: {results['coverage_percent']:.2f}%")
    print(f"🌱 Vegetation Area: {results['vegetation_area']:.2f} m² ({results['vegetation_area']/10000:.2f} ha)")
    print(f"🏫 Campus Total Area: {results['campus_area']:.2f} m² ({results['campus_area']/10000:.2f} ha)")
    print("=" * 60)
    
    # 可视化
    visualize_results(img, campus_mask, vegetation_mask, red_mask, results, output_name)
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
