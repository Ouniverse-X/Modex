# `final.tex` 附录代码图片化说明

- 仅修改论文附录中的代码呈现方式，正文内容未作改写。
- 附录代码先按原排版生成母版，再以 300 dpi 截取第 30--69 页，共 40 张。
- 图片已裁除母版 PDF 的页边距、页眉和页码，仅保留代码版心。
- 根目录 `final.tex` 中保留原始代码文本，但置于 `comment` 环境中，不参与排版；实际排版使用 `code_pages/code_page_01.png` 至 `code_page_40.png`。
- `output/final.pdf` 是在本目录独立编译得到的核验稿。因正文部分若干原图缺失，核验稿中相应位置使用占位框；这不影响附录代码图片。

编译核验命令：

```bash
cd /home/beihang/projects/Modex/final_build
TEXINPUTS='../Paper/ShuMoGoGoGo//:' latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=output final.tex
```
