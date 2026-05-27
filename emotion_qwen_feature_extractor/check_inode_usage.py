# -*- coding: utf-8 -*-
"""
文件名：check_inode_usage.py
功能：inode使用量检查与清理工具
创建时间：2025-07-11

本工具用于检查AutoDL平台的inode使用情况，
在实验开始前确保剩余inode足够，避免"磁盘空间不足"错误。
注意：AutoDL平台单实例限制200000个inode，超过此限制会报错。
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Tuple


def check_inode_usage(directory: str = ".") -> Tuple[int, int, int, float]:
    """
    检查指定目录所在分区的inode使用情况。

    Args:
        directory: 要检查的目录路径

    Returns:
        (used, total, available, usage_percentage): 已用、总计、可用inode数和百分比
    """
    try:
        result = subprocess.run(
            ["df", "-i", directory],
            capture_output=True,
            text=True,
            timeout=10
        )
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 5:
                total = int(parts[1])
                used = int(parts[2])
                available = int(parts[3])
                usage_pct = float(parts[4].rstrip('%'))
                return used, total, available, usage_pct
    except Exception as e:
        print(f"⚠ 检查inode失败: {e}")
    
    return 0, 0, 0, 0.0


def scan_directory_file_count(directory: str) -> Tuple[int, int]:
    """
    递归扫描目录，统计文件和目录数量。

    Args:
        directory: 目标目录

    Returns:
        (file_count, dir_count): 文件和目录数量
    """
    file_count = 0
    dir_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)
        dir_count += len(dirs)
    return file_count, dir_count


def find_large_directories(base_dir: str, top_n: int = 10) -> list:
    """
    查找文件数量最多的目录（inode消耗大户）。

    Args:
        base_dir: 基础目录
        top_n: 返回前N个目录

    Returns:
        [(dir_path, file_count), ...] 按文件数量降序排列
    """
    dir_fcount = []
    for root, dirs, files in os.walk(base_dir):
        dir_fcount.append((root, len(files)))
    dir_fcount.sort(key=lambda x: x[1], reverse=True)
    return dir_fcount[:top_n]


def safe_cleanup_pycache(base_dir: str, dry_run: bool = True) -> int:
    """
    安全清理__pycache__目录（Python缓存文件消耗大量inode）。

    Args:
        base_dir: 基础目录
        dry_run: 如果为True，仅打印将要删除的内容，不实际删除

    Returns:
        将要/已删除的文件数量
    """
    total_removed = 0
    for root, dirs, files in os.walk(base_dir):
        if "__pycache__" in dirs:
            pycache_dir = os.path.join(root, "__pycache__")
            fcount = len(list(Path(pycache_dir).rglob("*.pyc")))
            if not dry_run:
                import shutil
                shutil.rmtree(pycache_dir)
            print(f"  {'[DRY RUN] 将' if dry_run else '已'}删除: {pycache_dir} ({fcount}个文件)")
            total_removed += fcount
    return total_removed


def main():
    parser = argparse.ArgumentParser(
        description="inode使用量检查与清理工具 - 用于AutoDL平台inode限制监控"
    )
    parser.add_argument(
        "--check",
        type=str,
        default=".",
        help="检查指定目录所在分区的inode使用量"
    )
    parser.add_argument(
        "--scan",
        type=str,
        default=None,
        help="扫描指定目录，统计文件数量分布"
    )
    parser.add_argument(
        "--clean",
        type=str,
        default=None,
        help="清理指定目录下的__pycache__和临时文件"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="实际执行清理（不加此参数仅预览）"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("inode使用量检查工具")
    print("=" * 60)

    # 检查inode
    if args.check:
        used, total, available, pct = check_inode_usage(args.check)
        print(f"\n📊 inode状态 ({args.check}):")
        print(f"  总计: {total:,}")
        print(f"  已用: {used:,}")
        print(f"  可用: {available:,}")
        print(f"  使用率: {pct:.1f}%")

        # 警告阈值
        if total > 0:
            if pct > 90:
                print(f"  ⚠ 警告: inode使用率超过90%！")
            if pct > 80:
                print(f"  ⚠ 提醒: inode使用率超过80%，建议清理")
            if available < 100000:
                print(f"  ⚠ 提醒: 剩余inode不足100000个，可能触发AutoDL平台限制！")
            if available < 50000:
                print(f"  🚨 危险: 剩余inode严重不足！请立即清理！")

    # 扫描文件数量
    if args.scan:
        print(f"\n🔍 扫描目录: {args.scan}")
        file_count, dir_count = scan_directory_file_count(args.scan)
        print(f"  文件总数: {file_count:,}")
        print(f"  目录总数: {dir_count:,}")
        print(f"  总inode消耗: {file_count + dir_count:,}")

        # 查找最大的目录
        print(f"\n📁 文件数量最多的10个目录:")
        top_dirs = find_large_directories(args.scan, top_n=10)
        for dir_path, fcount in top_dirs:
            if fcount > 0:
                print(f"  {fcount:6d} 个文件 -> {dir_path}")

    # 清理__pycache__
    if args.clean:
        dry_run = not args.execute
        print(f"\n🧹 {'[预览模式] ' if dry_run else ''}清理: {args.clean}")
        removed = safe_cleanup_pycache(args.clean, dry_run=dry_run)
        if dry_run:
            print(f"\n  预览: 将删除 {removed} 个文件")
            print(f"  加上 --execute 参数实际执行清理")
        else:
            print(f"\n  已删除 {removed} 个文件")

    # 最终建议
    print("\n" + "=" * 60)
    print("💡 建议:")
    print("  1. 特征输出目录放在inode充足的分区")
    print("  2. 所有预测结果合并为单个JSON文件")
    print("  3. 所有日志合并为单个文件")
    print("  4. 实验后及时删除临时文件")
    print("=" * 60)


if __name__ == "__main__":
    main()
