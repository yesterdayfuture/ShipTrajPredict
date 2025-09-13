# Python argparse 命令行参数解析详解

`argparse` 是 Python 标准库中用于解析命令行参数的模块，功能强大且易于使用。下面我将详细介绍如何使用 `argparse`，包括参数传递、简写形式以及一些高级用法。

## 1. 基本用法

### 1.1 最简单的示例

```python
import argparse

# 创建解析器
parser = argparse.ArgumentParser(description='这是一个简单的示例程序')

# 添加参数
parser.add_argument('filename', help='输入文件名')

# 解析参数
args = parser.parse_args()

# 使用参数
print(f"处理文件: {args.filename}")
```

运行方式：
```bash
python script.py data.txt
```

### 1.2 添加可选参数

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', help='增加输出详细程度', action='store_true')
args = parser.parse_args()

if args.verbose:
    print("详细模式已开启")
```

运行方式：
```bash
python script.py --verbose
```

## 2. 参数简写（短选项）

`argparse` 允许为参数定义简写形式：

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', help='增加输出详细程度', action='store_true')
parser.add_argument('-o', '--output', help='输出文件名')
args = parser.parse_args()

if args.verbose:
    print("详细模式已开启")
if args.output:
    print(f"输出到文件: {args.output}")
```

运行方式：
```bash
python script.py -v -o result.txt
# 或者
python script.py --verbose --output result.txt
```

## 3. 常用参数类型

### 3.1 位置参数 vs 可选参数

```python
import argparse

parser = argparse.ArgumentParser()
# 位置参数（必须提供）
parser.add_argument('input_file', help='输入文件')
# 可选参数
parser.add_argument('--output', '-o', help='输出文件')
args = parser.parse_args()

print(f"输入文件: {args.input_file}")
if args.output:
    print(f"输出文件: {args.output}")
```

### 3.2 指定参数类型

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--count', type=int, help='重复次数')
parser.add_argument('--price', type=float, help='商品价格')
args = parser.parse_args()

if args.count:
    print(f"将重复 {args.count} 次")
if args.price:
    print(f"价格为 {args.price:.2f}")
```

### 3.3 布尔标志

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--enable', action='store_true', help='启用功能')
parser.add_argument('--disable', action='store_false', help='禁用功能')
args = parser.parse_args()

print(f"功能状态: {'启用' if args.enable else '禁用'}")
```

## 4. 高级用法

### 4.1 多值参数

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--numbers', nargs='+', type=int, help='一组数字')
args = parser.parse_args()

if args.numbers:
    print(f"数字总和: {sum(args.numbers)}")
```

运行方式：
```bash
python script.py --numbers 1 2 3 4 5
```

### 4.2 互斥参数

```python
import argparse

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('--add', action='store_true', help='添加模式')
group.add_argument('--delete', action='store_true', help='删除模式')
args = parser.parse_args()

if args.add:
    print("添加模式")
elif args.delete:
    print("删除模式")
else:
    print("请指定模式 (--add 或 --delete)")
```

### 4.3 子命令（类似 git commit, git push）

```python
import argparse

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='command')

# 创建子命令 'init'
parser_init = subparsers.add_parser('init', help='初始化项目')
parser_init.add_argument('--name', required=True, help='项目名称')

# 创建子命令 'build'
parser_build = subparsers.add_parser('build', help='构建项目')
parser_build.add_argument('--debug', action='store_true', help='调试模式')

args = parser.parse_args()

if args.command == 'init':
    print(f"初始化项目: {args.name}")
elif args.command == 'build':
    print(f"构建项目{' (调试模式)' if args.debug else ''}")
else:
    parser.print_help()
```

运行方式：
```bash
python script.py init --name myproject
python script.py build --debug
```

## 5. 默认值和必需参数

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--host', default='localhost', help='服务器地址')
parser.add_argument('--port', type=int, required=True, help='服务器端口')
args = parser.parse_args()

print(f"连接至 {args.host}:{args.port}")
```

## 6. 参数组和帮助信息美化

```python
import argparse

parser = argparse.ArgumentParser(description='一个高级示例程序')

# 创建参数组
input_group = parser.add_argument_group('输入选项')
input_group.add_argument('--input', required=True, help='输入文件')
input_group.add_argument('--format', choices=['json', 'csv'], help='输入格式')

output_group = parser.add_argument_group('输出选项')
output_group.add_argument('--output', help='输出文件')
output_group.add_argument('--overwrite', action='store_true', help='覆盖已存在文件')

args = parser.parse_args()

print(f"处理输入文件: {args.input} (格式: {args.format})")
if args.output:
    print(f"输出到: {args.output} {'(覆盖)' if args.overwrite else ''}")
```

## 总结

1. 使用 `ArgumentParser` 创建解析器
2. 使用 `add_argument()` 添加参数，可以定义短选项（如 `-v`）和长选项（如 `--verbose`）
3. 支持多种参数类型：布尔标志、整数、浮点数、字符串等
4. 可以设置默认值（`default`）和必需参数（`required=True`）
5. 支持多值参数（`nargs`）和互斥参数组
6. 支持子命令模式，适合复杂 CLI 工具
7. 可以分组参数美化帮助信息

通过合理使用这些功能，你可以创建出既强大又用户友好的命令行界面。