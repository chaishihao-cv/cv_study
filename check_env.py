"""
环境自查脚本（跑一次就知道当前解释器对不对）

用法：在 VS Code 里打开本文件，点右上角三角形运行，
     或者终端里 D:\\Anaconda\\anaconda3\\python.exe check_env.py
"""

import sys
import os

print("=" * 56)
print("1. 当前 Python 解释器")
print("=" * 56)
print("  sys.executable :", sys.executable)
print("  版本           :", sys.version.split()[0])
print("  PYTHONNOUSERSITE :", os.environ.get("PYTHONNOUSERSITE", "未设置（正常）"))

print()
print("=" * 56)
print("2. 关键包")
print("=" * 56)
try:
    import numpy
    print("  numpy      :", numpy.__version__)
except ImportError:
    print("  numpy      : ❌ 未安装")

try:
    import matplotlib
    print("  matplotlib :", matplotlib.__version__)
except ImportError:
    print("  matplotlib : ❌ 未安装")

try:
    import torch
    print("  torch      :", torch.__version__)
    print("  torchvision:", end=" ")
    try:
        import torchvision
        print(torchvision.__version__)
    except ImportError:
        print("❌ 未安装")
    print("  CUDA 可用  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  GPU        :", torch.cuda.get_device_name(0))
        print("  显存       : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))
    else:
        print("  ⚠️ CUDA 不可用（可能装的是 CPU 版 torch）")
except ImportError:
    print("  torch      : ❌ 未安装（或当前解释器看不到用户级 site-packages）")

print()
print("=" * 56)
print("3. 结论")
print("=" * 56)
ok = True
if "anaconda3" not in sys.executable.lower().replace("/", "\\"):
    print("  ⚠️ 解释器不是 D:\\Anaconda\\anaconda3\\python.exe，去右下角切换")
    ok = False
try:
    import torch
    if not torch.cuda.is_available():
        print("  ⚠️ CUDA 不可用")
        ok = False
except ImportError:
    ok = False

print("  ✅ 环境正常，可以开始练习" if ok else "  ❌ 有问题，按上面提示处理")
