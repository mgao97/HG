import subprocess

# 使用 nice 命令设置优先级
priority = -20  # 提高优先级
command = ["nice", "-n", str(priority), "python3", "hypergcn_cvae_generate_coauthorcora.py"]

# 启动进程并设置优先级
subprocess.run(command)
