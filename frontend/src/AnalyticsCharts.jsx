import React from "react";

const COLORS = ["#4267e8", "#8a63d2", "#16a59a", "#e69b43", "#d85d79"];
const pct = n => `${(Number(n) * 100).toFixed(1)}%`;
const modelText = s => ({LogisticRegression:"Logistic Regression",RandomForest:"Random Forest"}[s] || s);

export function DonutChart({ data, center, label, size = 190 }) {
  const total = data.reduce((sum, item) => sum + Number(item.value), 0) || 1;
  const r = 62, circ = 2 * Math.PI * r;
  let offset = 0;
  return <div className="donut-wrap"><svg viewBox="0 0 180 180" role="img" aria-label={label} width={size} height={size}>
    <circle cx="90" cy="90" r={r} fill="none" stroke="#e9edf5" strokeWidth="18"/>
    {data.map((item, i) => { const value = Number(item.value); const dash = circ * value / total; const node = <circle key={item.label} cx="90" cy="90" r={r} fill="none" stroke={item.color || COLORS[i % COLORS.length]} strokeWidth="18" strokeDasharray={`${dash} ${circ-dash}`} strokeDashoffset={-offset} transform="rotate(-90 90 90)"><title>{item.label}: {pct(value/total)}</title></circle>; offset += dash; return node; })}
    <text x="90" y="85" textAnchor="middle" className="donut-center">{center}</text><text x="90" y="105" textAnchor="middle" className="donut-sub">{label}</text>
  </svg><div className="donut-legend">{data.map((item,i)=><div key={item.label}><i style={{background:item.color||COLORS[i%COLORS.length]}}/><span>{item.label}</span><b>{pct(Number(item.value)/total)}</b></div>)}</div></div>;
}

export function GroupedBars({ rows, metrics, title }) {
  const W=760,H=300,L=48,R=14,T=22,B=75, plotW=W-L-R, plotH=H-T-B, groups=Math.max(rows.length,1), groupW=plotW/groups, gap=3, barW=Math.min(20,(groupW-12)/metrics.length-gap);
  return <div className="chart-scroll"><svg className="chart-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
    {[0,.25,.5,.75,1].map(v=><g key={v}><line x1={L} x2={W-R} y1={T+plotH*(1-v)} y2={T+plotH*(1-v)} className="chart-grid"/><text x={L-9} y={T+plotH*(1-v)+4} textAnchor="end" className="chart-axis">{Math.round(v*100)}%</text></g>)}
    {rows.map((row,ri)=>{const base=L+ri*groupW+(groupW-metrics.length*(barW+gap))/2;return <g key={row.model}>{metrics.map((metric,mi)=>{const value=Math.max(0,Math.min(1,Number(row[metric.key])||0));const h=plotH*value;return <g key={metric.key}><rect x={base+mi*(barW+gap)} y={T+plotH-h} width={barW} height={h} rx="4" fill={metric.color}><title>{modelText(row.model)} · {metric.label}: {pct(value)}</title></rect></g>})}<text transform={`translate(${L+ri*groupW+groupW/2},${T+plotH+17}) rotate(-22)`} textAnchor="end" className="chart-label">{modelText(row.model)}</text></g>})}
  </svg><div className="chart-legend">{metrics.map(metric=><span key={metric.key}><i style={{background:metric.color}}/>{metric.label}</span>)}</div></div>;
}

export function HorizontalBars({ rows, labelKey, valueKey, title, valueFormat = pct }) {
  const max=Math.max(...rows.map(row=>Number(row[valueKey])||0),.00001);
  return <div className="hbar-chart" role="img" aria-label={title}>{rows.map((row,i)=>{const value=Number(row[valueKey]);const label=labelKey==="model"?modelText(row[labelKey]):row[labelKey];return <div className="hbar-row" key={row[labelKey]} title={`${label}: ${valueFormat(value)}`}><span className="hbar-name">{label}</span><div className="hbar-track"><i style={{width:`${value/max*100}%`,background:COLORS[i%COLORS.length]}}/></div><b>{valueFormat(value)}</b></div>})}</div>;
}

export function LineChart({ series, xLabels, title, valueFormat = pct }) {
  const W=760,H=290,L=48,R=20,T=24,B=60,pw=W-L-R,ph=H-T-B;
  return <div className="chart-scroll"><svg className="chart-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
    {[0,.25,.5,.75,1].map(v=><g key={v}><line x1={L} x2={W-R} y1={T+ph*(1-v)} y2={T+ph*(1-v)} className="chart-grid"/><text x={L-9} y={T+ph*(1-v)+4} textAnchor="end" className="chart-axis">{Math.round(v*100)}%</text></g>)}
    {series.map((s,i)=>{const pts=s.values.map((value,j)=>`${L+(xLabels.length===1?pw/2:j*pw/(xLabels.length-1))},${T+ph*(1-Math.max(0,Math.min(1,Number(value)||0)))}`).join(" ");return <g key={s.name}><polyline points={pts} fill="none" stroke={COLORS[i%COLORS.length]} strokeWidth="3" strokeLinejoin="round" strokeLinecap="round"/>{s.values.map((value,j)=><circle key={j} cx={L+(xLabels.length===1?pw/2:j*pw/(xLabels.length-1))} cy={T+ph*(1-Number(value))} r="4.5" fill={COLORS[i%COLORS.length]}><title>{s.name} · {xLabels[j]}: {valueFormat(value)}</title></circle>)}</g>})}
    {xLabels.map((label,i)=><text key={label} x={L+(xLabels.length===1?pw/2:i*pw/(xLabels.length-1))} y={H-23} textAnchor="middle" className="chart-label">{label}</text>)}
  </svg><div className="chart-legend">{series.map((s,i)=><span key={s.name}><i style={{background:COLORS[i%COLORS.length]}}/>{s.name}</span>)}</div></div>;
}

export function RocChart({ curves, models, featureSet }) {
  const W=420,H=330,L=48,R=18,T=16,B=43,pw=W-L-R,ph=H-T-B;
  return <div className="chart-scroll"><svg className="chart-svg roc-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`ROC curves for ${featureSet}`}>
    {[0,.25,.5,.75,1].map(v=><g key={v}><line x1={L} x2={W-R} y1={T+ph*(1-v)} y2={T+ph*(1-v)} className="chart-grid"/><line x1={L+pw*v} x2={L+pw*v} y1={T} y2={T+ph} className="chart-grid"/><text x={L-8} y={T+ph*(1-v)+4} textAnchor="end" className="chart-axis">{v.toFixed(1)}</text></g>)}
    <line x1={L} y1={T+ph} x2={L+pw} y2={T} stroke="#aeb8c9" strokeDasharray="5 5"/>
    {models.map((model,i)=>{const c=curves[`${model}|${featureSet}`];if(!c)return null;const points=c.fpr.map((x,j)=>`${L+pw*x},${T+ph*(1-c.tpr[j])}`).join(" ");return <polyline key={model} points={points} fill="none" stroke={COLORS[i%COLORS.length]} strokeWidth="2.5"><title>{model}: ROC-AUC {Number(c.auc).toFixed(3)}</title></polyline>})}
    <text x={L+pw/2} y={H-5} textAnchor="middle" className="chart-axis">False positive rate</text><text transform={`translate(12 ${T+ph/2}) rotate(-90)`} textAnchor="middle" className="chart-axis">True positive rate</text>
  </svg><div className="chart-legend">{models.map((model,i)=>{const c=curves[`${model}|${featureSet}`];return <span key={model}><i style={{background:COLORS[i%COLORS.length]}}/>{modelText(model)}{c?` · ${Number(c.auc).toFixed(3)}`:""}</span>})}</div></div>;
}

export function ConfusionHeatmap({ matrix }) {
  if(!matrix)return <p className="muted">Confusion matrix is unavailable.</p>;
  const max=Math.max(...matrix.flat(),1);
  return <div className="heatmap-wrap"><div className="heatmap-axis-title">Predicted</div><div className="heatmap"><div/><b>Not placed</b><b>Placed</b><b>Not placed</b>{matrix[0].map((v,i)=><div key={`a${i}`} className="heat-cell" style={{background:`rgba(66,103,232,${.1+.75*v/max})`}}><strong>{v.toLocaleString()}</strong><small>Actual not placed</small></div>)}<b>Placed</b>{matrix[1].map((v,i)=><div key={`b${i}`} className="heat-cell" style={{background:`rgba(22,165,154,${.1+.75*v/max})`}}><strong>{v.toLocaleString()}</strong><small>Actual placed</small></div>)}</div><div className="heatmap-foot">Rows are actual labels · columns are model predictions</div></div>;
}
