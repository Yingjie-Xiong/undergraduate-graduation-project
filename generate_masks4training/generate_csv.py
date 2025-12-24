import pandas as pd
import os

def main(args):
    # 获取已掩码的文件
    masked_files = [args.llama_masked_outdir + x for x in os.listdir(args.llama_masked_outdir) if "_mask" in x]
    # 获取未掩码的文件
    non_masked_files = ["_".join(x.split("_")[:-1]) + "." + x.split(".")[-1] for x in masked_files]
    print(masked_files, non_masked_files)
    # 确保掩码文件和未掩码文件的数量一致
    assert len(masked_files) == len(non_masked_files)
    # 生成训练集和验证集
    total_files = len(non_masked_files)
    split_index = int(total_files * 0.8)  # 80%用于训练，20%用于验证

    train_files = non_masked_files[:split_index]
    train_masks = masked_files[:split_index]
    val_files = non_masked_files[split_index:]
    val_masks = masked_files[split_index:]
    df = pd.DataFrame(columns=["image_path", "mask_path", "partition"])

    # 填充训练集
    train_partitions = ["train"] * len(train_files)
    df_train = pd.DataFrame({
        "image_path": train_files,
        "mask_path": train_masks,
        "partition": train_partitions
    })

    # 填充验证集
    val_partitions = ["validation"] * len(val_files)
    df_val = pd.DataFrame({
        "image_path": val_files,
        "mask_path": val_masks,
        "partition": val_partitions
    })

    # 合并数据集
    df = pd.concat([df_train, df_val], ignore_index=True)

    # 保存CSV文件
    df.to_csv(args.csv_out_path, index=False)


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('--llama_masked_outdir', type=str, help='数据集生成的配置路径')
    aparser.add_argument('--csv_out_path', type=str, default='output.csv', help='输出CSV文件路径')
    main(aparser.parse_args())