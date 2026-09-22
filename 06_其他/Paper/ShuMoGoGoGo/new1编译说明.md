# new1 编译说明

本次以 `new1.tex` 正文为依据重写摘要，保存为 `A题摘要_new1.md`，并同步替换源文件中的旧摘要及关键词。未重新进行数值计算。第三问采用正文报告值 **57.4724 h**（连续临界值向上保留四位小数）；第四问为 **51.0906 h**。新版正文未展开 Bessel 基准检验，因此此次摘要没有列入该项。

在此目录执行 `latexmk -xelatex -interaction=nonstopmode -halt-on-error new1.tex`，输出为 `build/new1.pdf`。

为在服务器编译，移除了 Windows 专用字体路径和 Inkscape 程序名，采用本机 Noto Sans CJK SC 与 TeX 自带 Fandol 字体。`new1_graphics_compat.tex` 优先读取同名 PDF/PNG；找不到的图显示“插图文件缺失，待补齐”，并记录编译警告。源文件仍保留这些缺失图的原引用名，补齐后可重新编译。仅有 SVG 的图片还需要可用的 Inkscape 及对应导出配置，提供同名 PDF/PNG 则可直接编译。

第四问长度反演图与半径对照图已分别关联本地 `problem4-2` 和 `problem4-3` 的现有结果图。

仍缺少以下 6 张不同的图片，其中等值线图在正文中引用两次：

- `figures/问题二新结果展示.svg`
- `figures/问题三结果展示折线图.png`
- `figures/问题三三维曲面图.png`
- `figures/问题三等值线图.svg`
- `figures/问题三结果展示圆柱.svg`
- `figures/问题四结果展示.svg`

当前 PDF 是带缺图标记的预览版。源文件标题仍为旧模板的“基于动态搜索的‘板凳龙’运动状态及路线研究”，本次摘要重写未代拟标题。补齐图片并更换标题后再用于最终提交。
