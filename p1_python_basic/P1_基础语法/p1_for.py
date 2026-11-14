name = "elsen"

num = 0

for e in name:
    if e == "e": num += 1

print(num)

# # ... existing code ...

# for 循环的其他写法

# 1. range：按次数/数字区间循环
for i in range(3):  # 0, 1, 2
    print(i)

for i in range(1, 4):  # 1, 2, 3（左闭右开）
    print(i)

for i in range(0, 6, 2):  # 0, 2, 4（步长为 2）
    print(i)

for i in range(3, 0, -1):  # 3, 2, 1（倒序）
    print(i)

# 2. enumerate：同时获取下标和元素
for index, e in enumerate(name):
    print(index, e)

for index, e in enumerate(name, start=1):  # 下标从 1 开始
    print(index, e)

# 3. 遍历字典
d = {"a": 1, "b": 2}

for key in d:  # 默认遍历 key
    print(key)

for value in d.values():  # 遍历 value
    print(value)

for key, value in d.items():  # 同时遍历 key 和 value（解包）
    print(key, value)

# 4. zip：并行遍历多个序列
names = ["elsen", "bob"]
ages = [18, 20]

for n, a in zip(names, ages):
    print(n, a)

# 5. 列表推导式：一行生成新列表
squares = [x ** 2 for x in range(5)]
evens = [x for x in range(10) if x % 2 == 0]
print(squares, evens)

# 6. break / continue / else
for i in range(5):
    if i == 2:
        continue  # 跳过本次，进入下一次循环
    if i == 4:
        break  # 提前结束整个循环
    print(i)
else:
    # 循环没有被 break 打断时才会执行
    print("循环正常结束")

# 7. 解包遍历
pairs = [(1, "a"), (2, "b")]
for num, char in pairs:
    print(num, char)

# 8. 遍历文件（逐行读取）
# with open("data.txt", encoding="utf-8") as f:
#     for line in f:
#         print(line.strip())


# 列表
list = [1, 2, 3, 4]
read_list = (1, 2, 3, 4)

# 字典
dist = {
    "name": "elsen",
    "age": 18
}

# 打印字典的key和value
for key in dist:
    print(key, dist[key])

