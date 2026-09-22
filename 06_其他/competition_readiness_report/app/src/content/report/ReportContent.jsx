import React from "react";

import {
  DataComponent, DataTable, MetricCard, ReportSection, RichNarrative, useDataApp,
} from "../../data-app-public.jsx";

const findingColumns = [
  { field: "priority", label: "优先级", presentation: "status" },
  { field: "area", label: "领域" },
  { field: "finding", label: "审计发现" },
  { field: "impact", label: "影响" },
  { field: "nextStep", label: "必须动作" },
];

const verificationColumns = [
  { field: "check", label: "检查项" },
  { field: "result", label: "结论", presentation: "status" },
  { field: "value", label: "证据" },
];

const precisionColumns = [
  { field: "question", label: "问题" },
  { field: "field", label: "关键场" },
  { field: "errorBound", label: "误差上界", renderCell: (value) => Number(value).toExponential(3) },
  { field: "fourDecimalHalfUnit", label: "四位小数半单位", renderCell: (value) => Number(value).toExponential(1) },
  { field: "thresholdUsedPct", label: "误差预算占用", renderCell: (value) => `${Number(value).toFixed(2)}%` },
];

const sensitivityColumns = [
  { field: "question", label: "问题" },
  { field: "assumption", label: "仅改变的假设" },
  { field: "timeDifferenceMinutes", label: "时长变化", renderCell: (value) => `${Number(value).toFixed(2)} min` },
  { field: "numericalTimeUncertaintySeconds", label: "数值时刻误差", renderCell: (value) => `${Number(value).toFixed(4)} s` },
  { field: "ratioToNumericalUncertainty", label: "假设影响 / 数值误差", renderCell: (value) => `${Math.round(value).toLocaleString()}×` },
];

export function ReportContent() {
  const { reviewedPeriodRows, visible, canEdit, mode, appTitle, setAppTitle } = useDataApp();
  const findings = reviewedPeriodRows("readiness_findings");
  const p0 = findings.filter((row) => row.priority === "P0");
  const p1 = findings.filter((row) => row.priority === "P1");
  const verification = reviewedPeriodRows("verification_checks");
  const precision = reviewedPeriodRows("numerical_headroom");
  const sensitivity = reviewedPeriodRows("scenario_sensitivity");

  return <article className="report-content" aria-label="A题建模与竞赛论文预审报告">
    <header className="report-hero">
      <div className="verdict-row">
        <span className="verdict-badge">需要修订 · 不宜直接提交</span>
        <span className="as-of">审计截至 2026-09-11</span>
      </div>
      <h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning
        aria-label={canEdit && mode === "edit" ? "编辑报告标题" : undefined}
        onBlur={canEdit && mode === "edit" ? (event) => setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}
        onKeyDown={canEdit && mode === "edit" ? (event) => {
          if (event.key === "Enter") { event.preventDefault(); event.currentTarget.blur(); }
        } : undefined}>{appTitle}</h1>
      <RichNarrative id="report:intro" className="report-deck" label="编辑报告导语"
        value="结论先行：**四问求解器的数值验证已接近论文级**，但当前目录仍不是可提交的国赛论文。决定竞争力的下一步，不是继续盲目加密网格，而是补齐 Q4 物理守恒、Q3 严格阈值、模型敏感性、唯一结果口径、AI 合规与正式论文成品。" />
    </header>

    <div className="report-facts" aria-label="核心审计指标">
      {visible("metric-p0") && <MetricCard id="metric-p0" title="提交阻断项" queryId="readiness_findings"
        sourceRows={p0} value={`${p0.length} 项 P0`} comparison="全部需在定稿前关闭" negative
        description="任何一项遗留，都可能造成结论不严谨、材料不一致或竞赛合规风险。" />}
      {visible("metric-tests") && <MetricCard id="metric-tests" title="自动化测试" queryId="verification_checks"
        sourceRows={verification} value="27 / 27" comparison="本次复跑全部通过"
        description="另完成 Python 编译检查、Bessel 二阶验证和 BDF–Radau 对照。" />}
      {visible("metric-grid") && <MetricCard id="metric-grid" title="四位小数误差预算" queryId="numerical_headroom"
        sourceRows={precision} value="≤ 29.11%" comparison="四问关键含水率场均留有余量"
        description="最大空间离散误差上界占四位小数半单位的比例；这是数值精度，不是模型真实性。" />}
      {visible("metric-assumption") && <MetricCard id="metric-assumption" title="单一假设可改变量" queryId="scenario_sensitivity"
        sourceRows={sensitivity} value="18.28 min" comparison="远大于亚秒级数值误差" negative
        description="仅改变 4 h 后环境延拓口径，Q3 的烘干时长变化。" />}
    </div>

    {visible("executive-summary") && <ReportSection id="executive-summary" title="总判断"
      queryId="readiness_findings" queryIds={["readiness_findings", "verification_checks"]}
      sourceRowsByQuery={{ readiness_findings: findings, verification_checks: verification }} showHeading={false}>
      <RichNarrative id="executive-summary:body" label="编辑总判断" value={`## 总判断

- **可保留的主干**：圆柱径向 Fourier–Fick 耦合框架、有限体积守恒离散、刚性积分器、网格/时间收敛与 Excel 逐格核验。
- **必须返工的核心**：Q4 中收缩体积、密度经验式、干基含水率之间的守恒关系；Q3 必须从临界等号时刻改为严格低于阈值的安全时刻。
- **竞争力缺口**：目前证明了“程序把所写方程解得很准”，尚未充分证明“方程足够可信、结论对关键假设稳健”。
- **提交缺口**：没有最终 PDF/Word、AI 使用声明与详情 PDF、匿名且小于 20 MB 的支撑包，也没有冻结唯一的 result1.xlsx–result4.xlsx。`} />
    </ReportSection>}

    {visible("verified-strengths") && <section className="report-section">
      <RichNarrative id="verified-strengths:intro" label="编辑数值验证说明" value={`## 已经达到论文级的部分

这是项目目前最强的一层：固定域 Bessel 基准表现出约二阶空间收敛，BDF 与 Radau 的场值差小于 2×10⁻⁶，四问测试全部通过。Q1–Q4 的关键含水率误差上界均小于四位小数半单位 5×10⁻⁵。`} />
      <DataComponent id="verification-table" title="复跑与交叉验证结果" queryId="verification_checks"
        kind="table" displayRows={verification} sourceRows={verification}
        description="本次预审在项目现有 Conda Python 环境中复跑。">
        <DataTable rows={verification} columns={verificationColumns} searchable={false} rowKey="check"
          caption="数值实现验证检查表" />
      </DataComponent>
      <DataComponent id="precision-table" title="四位小数精度余量" queryId="numerical_headroom"
        kind="table" displayRows={precision} sourceRows={precision}
        description="误差预算占用越低，距 5×10⁻⁵ 的四舍五入临界值越远。">
        <DataTable rows={precision} columns={precisionColumns} searchable={false} rowKey="question"
          caption="四问关键场的误差上界" />
      </DataComponent>
      <RichNarrative id="verified-strengths:caveat" className="report-disclosure" label="编辑验证边界"
        value="重要边界：Q2/Q4 的 Radau 对照复用了同一半离散方程，因此主要验证时间积分器，不能独立排除共同的移动边界变换错误；固定域 Bessel 解也没有覆盖时变半径。" />
    </section>}

    {visible("blocking-findings") && <section className="report-section">
      <RichNarrative id="blocking-findings:intro" label="编辑阻断项说明" value={`## 五项 P0：定稿前必须全部关闭

其中 **Q4 物理一致性** 是科学性最高风险，**AI 声明与论文格式** 是提交资格风险，**Q3 严格阈值与结果口径** 是答案正确性风险。`} />
      <DataComponent id="p0-table" title="P0 阻断清单" queryId="readiness_findings"
        kind="table" displayRows={p0} sourceRows={p0}
        description="按影响优先处理；表内每项都附有可执行的关闭动作。">
        <DataTable rows={p0} columns={findingColumns} searchable={false} rowKey="area" caption="P0 阻断项" />
      </DataComponent>
    </section>}

    {visible("model-risk-summary") && <section className="report-section">
      <RichNarrative id="model-risk-summary:intro" label="编辑模型风险说明" value={`## 模型误差远大于数值误差

继续将网格加密一倍，已不是最有价值的投入。只改变附件 1 结束后的环境延拓方式，Q3、Q4 的结束时刻就分别变化约 18.28 min 和 15.98 min；相对于当前数值时刻误差，其影响达到约 7.7×10⁴ 倍和 2.9×10⁴ 倍。下一轮算力应优先用于**物理假设敏感性与反事实实验**。`} />
      <DataComponent id="sensitivity-table" title="假设不确定性与数值误差的量级对比" queryId="scenario_sensitivity"
        kind="table" displayRows={sensitivity} sourceRows={sensitivity}
        description="两行均是在其他离散配置相同的情况下，仅改变长期边界口径。">
        <DataTable rows={sensitivity} columns={sensitivityColumns} searchable={false} rowKey="question"
          caption="长期环境边界敏感性" />
      </DataComponent>
    </section>}

    {visible("competitiveness-gaps") && <section className="report-section">
      <RichNarrative id="competitiveness-gaps:intro" label="编辑竞争力缺口" value={`## 从“完整”到“高竞争力”还差什么

P1 不是格式润色，而是决定论文说服力的科学工作。优先顺序建议为：**移动边界守恒验证 → 参数/假设敏感性 → 2×2 因素分解 → 结果可视化 → 外部有效性证据 → 可复现与匿名打包**。`} />
      <DataComponent id="p1-table" title="P1 竞争力提升清单" queryId="readiness_findings"
        kind="table" displayRows={p1} sourceRows={p1}
        description="完成这些项目后，论文才能从可靠计算报告升级为有辨识度的竞赛论文。">
        <DataTable rows={p1} columns={findingColumns} searchable={false} rowKey="area" caption="P1 竞争力提升项" />
      </DataComponent>
    </section>}

    {visible("sprint-plan") && <ReportSection id="sprint-plan" title="推荐冲刺路线"
      queryId="readiness_findings" sourceRows={findings} showHeading={false}>
      <RichNarrative id="sprint-plan:body" label="编辑冲刺路线" value={`## 推荐冲刺路线

1. **先冻结口径**：明确 Q3/Q4 在 4 h 后统一采用 50 °C、0.05 kg/kg；整理唯一的 result1.xlsx–result4.xlsx，并生成哈希清单。
2. **关闭答案漏洞**：重算 Q3 安全结束时刻，使连续场含水率上界及四位小数展示都严格低于 0.15；同步正文和结果表。
3. **重建 Q4 守恒闭环**：从材料坐标 Jacobian 出发明确 ρ、C 与干固体质量的定义；补时变 R(t) 制造解或独立 ALE 对照。
4. **把算力用在模型而非网格**：对 h、hₘ、潜热/焓项、端面换热换质、长期边界、R(t) 插值与物性外推做敏感性；至少给出排序和结论区间。
5. **做 2×2 反事实**：固定/收缩半径 × 附录 3/附录 4 物性，拆分“收缩”和“物性变化”对时长的贡献，形成论文的核心洞见。
6. **重写正式论文**：一页摘要与关键词、答案前置、主文控制在 30 页内且无目录；补时空热图、径向剖面、中心/表面轨迹、R(t)、收敛图及局限性。
7. **完成提交治理**：按 [2026 论文格式规范](https://www.mcm.edu.cn/html_cn/node/4cd596519c9eb9fbd866398f6df0caa3.html) 和 [2026 AI 工具使用规定](https://www.mcm.edu.cn/html_cn/node/fef94648f2836ab6cc81586f4c38512b.html) 生成匿名、小于 20 MB 的论文与支撑材料，并逐项人工核验 AI 参与内容。`} />
    </ReportSection>}

    {visible("final-verdict") && <div className="final-verdict">
      <RichNarrative id="final-verdict:body" label="编辑最终结论" value={`## 最终结论

当前状态可概括为：**数值实现强，论文工程未完成，物理可信度仍有一个核心缺口。** 若先关闭五项 P0，再完成 Q4 守恒验证、敏感性与 2×2 因素分解，这套工作具备冲击高质量国赛论文的基础；若只继续细化网格而不处理这些问题，边际收益很低。`} />
    </div>}
  </article>;
}
