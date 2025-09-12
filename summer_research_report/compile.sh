#!/bin/bash

# 编译 LaTeX 主文件 main.tex 并生成参考文献
# 用法: ./compile.sh

set -e  # 出错就停止执行

# 关闭 Skim（macOS 下的 PDF 阅读器，否则会锁文件）
killall Skim 2>/dev/null || true

# LaTeX 编译流程
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex

echo "✅ Compilation finished. PDF generated: main.pdf"
