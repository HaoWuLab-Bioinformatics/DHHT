import subprocess
import os
CUDA_LAUNCH_BLOCKING=1


def run_experiment():
    # 显式指定 Conda 解释器
    python_executable = "/home/mzy/anaconda3/envs/pytorch/bin/python3.10"

    # 设置工作目录到 medium
    os.chdir("/home/mzy/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/large")

    command = [
        python_executable, "main.py",
        "--dataset", "amazon2m",
        "--method", "GCN",
        "--lr", "0.005",
        "--weight_decay", "0.0",
        "--hidden_channels", "64",
        "--use_graph", "1",
        "--gnn_dropout", "0.",
        "--gnn_use_bn", "1",
        "--gnn_num_layers", "3",
        "--gnn_use_init", "1",
        "--trans_num_layers", "1",
        "--trans_use_residual", "1",
        "--trans_use_bn", "0",
        "--graph_weight", "0.2",
        "--trans_dropout", "0.2",
        "--device", "0",  # 使用GPU 1
        "--runs", "1",  # 只进行一次实验
        "--power_k", "2.0",  # power_k 设置为 2.0

        "--decoder", "hyp",  # 解码器使用 hyp
        "--k_in", "2",  # 输入层的曲率为 1.0
        "--k_out", "0.5",  # 输出层的曲率为 2.0
        "--data_dir", "../data",  # 数据集所在路径
        "--decoder_type", "hyp",  # 解码器类型使用 hyp
        "--sub_dataset", "gcn_data",  # 子数据集选择 gcn_data

        "--protocol", "semi",  # 使用 semi 监督协议
        "--rand_split", "1",  # 不使用随机分割
        "--display_step", "1",  # 每50步输出一次信息


        "--optimizer_type", "adam",  # 使用 Adam 优化器
        "--hyp_optimizer_type", "radam",  # 使用 RAdam 优化器
        "--patience", "100",  # 早停的耐心度

    ]

    # 运行 main.py，实时读取输出
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    # 逐行打印输出，避免缓存
    for line in process.stdout:
        print(line, end="")  # 让输出实时刷新
    for line in process.stderr:
        print(line, end="")  # 也打印错误信息

    process.stdout.close()
    process.stderr.close()
    process.wait()  # 等待进程结束


if __name__ == "__main__":
    run_experiment()