import time

def cpu_stress_operation(operations_count):
    # 一个CPU密集型操作，比如大量的循环计算
    total = 0
    for i in range(operations_count):
        total += i**2
    return total

if __name__ == "__main__":
    operations_count = 10**8  # 你可以调整操作的次数

    print("Starting CPU stress test")
    start_time = time.time()

    # 执行CPU密集型操作
    cpu_stress_operation(operations_count)

    end_time = time.time()
    print(f"Finished CPU stress test in {end_time - start_time} seconds")
