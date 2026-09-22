#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件工具模块
提供文件读写相关的通用工具函数
"""

import os
import json
from typing import List, Dict, Any

def load_txt(file_path: str, encoding: str = 'utf-8') -> List[str]:
    """
    读取txt文件内容，返回行列表
    
    Args:
        file_path: 文件路径
        encoding: 文件编码，默认utf-8
        
    Returns:
        List[str]: 文件内容行列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding=encoding) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    return lines

def load_json(file_path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    读取json文件内容
    
    Args:
        file_path: 文件路径
        encoding: 文件编码，默认utf-8
        
    Returns:
        Dict[str, Any]: JSON数据
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding=encoding) as f:
        data = json.load(f)
    
    return data

def dump_json(data: Any, file_path: str, encoding: str = 'utf-8', ensure_ascii: bool = False, indent: int = 2) -> None:
    """
    将数据写入json文件
    
    Args:
        data: 要写入的数据
        file_path: 文件路径
        encoding: 文件编码，默认utf-8
        ensure_ascii: 是否确保ASCII编码，默认False
        indent: 缩进空格数，默认2
    """
    # 确保目录存在
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    with open(file_path, 'w', encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent, default=str)

def load_csv_as_dict(file_path: str, encoding: str = 'utf-8') -> List[Dict[str, str]]:
    """
    读取CSV文件内容，返回字典列表
    
    Args:
        file_path: 文件路径
        encoding: 文件编码，默认utf-8
        
    Returns:
        List[Dict[str, str]]: CSV数据字典列表
    """
    import csv
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding=encoding) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 清理数据
            cleaned_row = {}
            for key, value in row.items():
                cleaned_row[key.strip()] = value.strip() if value else ''
            data.append(cleaned_row)
    
    return data

def get_file_extension(file_path: str) -> str:
    """
    获取文件扩展名
    
    Args:
        file_path: 文件路径
        
    Returns:
        str: 文件扩展名（包含点号）
    """
    return os.path.splitext(file_path)[1].lower()

def is_file_exists(file_path: str) -> bool:
    """
    检查文件是否存在
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 文件是否存在
    """
    return os.path.exists(file_path)

def get_file_size(file_path: str) -> int:
    """
    获取文件大小（字节）
    
    Args:
        file_path: 文件路径
        
    Returns:
        int: 文件大小（字节）
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    return os.path.getsize(file_path)

def list_files_in_directory(directory_path: str, extension: str = None) -> List[str]:
    """
    列出目录中的文件
    
    Args:
        directory_path: 目录路径
        extension: 文件扩展名过滤（可选）
        
    Returns:
        List[str]: 文件路径列表
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"目录不存在: {directory_path}")
    
    files = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            if extension is None or filename.lower().endswith(extension.lower()):
                files.append(file_path)
    
    return files

def create_directory_if_not_exists(directory_path: str) -> None:
    """
    如果目录不存在则创建目录
    
    Args:
        directory_path: 目录路径
    """
    os.makedirs(directory_path, exist_ok=True)

def read_file_lines(file_path: str, encoding: str = 'utf-8') -> List[str]:
    """
    读取文件所有行
    
    Args:
        file_path: 文件路径
        encoding: 文件编码
        
    Returns:
        List[str]: 文件行列表
    """
    with open(file_path, 'r', encoding=encoding) as f:
        return f.readlines()

def write_file_lines(file_path: str, lines: List[str], encoding: str = 'utf-8') -> None:
    """
    将行列表写入文件
    
    Args:
        file_path: 文件路径
        lines: 行列表
        encoding: 文件编码
    """
    # 确保目录存在
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    with open(file_path, 'w', encoding=encoding) as f:
        f.writelines(lines)
        
def save_md_to_file(md_content, file_path):
    """
    将 Markdown 内容保存到指定文件。
    md_content: Markdown 格式的字符串。
    file_path: 目标文件路径。
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Markdown 内容已保存到 {file_path}")

# 测试函数
def main():
    """主函数 - 用于测试"""
    print("文件工具模块测试")
    
    # 创建测试目录和文件
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # 测试txt文件读写
    txt_file = os.path.join(test_dir, "test.txt")
    test_lines = ["第一行", "第二行", "第三行"]
    write_file_lines(txt_file, [line + "\n" for line in test_lines])
    
    loaded_lines = load_txt(txt_file)
    print(f"TXT文件读取测试: {loaded_lines}")
    
    # 测试json文件读写
    json_file = os.path.join(test_dir, "test.json")
    test_data = {"name": "测试", "value": 123, "items": [1, 2, 3]}
    dump_json(test_data, json_file)
    
    loaded_data = load_json(json_file)
    print(f"JSON文件读取测试: {loaded_data}")
    
    # 清理测试文件
    import shutil
    shutil.rmtree(test_dir)
    print("测试完成，临时文件已清理")

if __name__ == "__main__":
    main()
