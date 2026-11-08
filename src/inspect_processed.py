
from pathlib import Path
import numpy as np

root = Path("data/processed")

files = [
    root / "skab_windows.npz",
    root / "normalization.npz",
]

for file in files:
    print("\n" + "=" * 60)
    print("检查文件：", file)
    print("=" * 60)

    if not file.exists():
        print("文件不存在！")
        continue

    with np.load(file, allow_pickle=False) as data:
        print("内部变量名称：", data.files)

        for name in data.files:
            arr = data[name]

            print("\n变量名称：", name)
            print("形状：", arr.shape)
            print("数据类型：", arr.dtype)

            if np.issubdtype(arr.dtype, np.number):
                print("是否全部为有限数值：", np.isfinite(arr).all())

                if arr.size > 0:
                    print("最小值：", np.min(arr))
                    print("最大值：", np.max(arr))
