"""Render the existing-capture ablation without altering trial results."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path)
args=parser.parse_args()
with (args.directory/'summary.csv').open(encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
lookup={(r['rgr_db'],r['pattern'],r['decoder']):r for r in rows}
text=['# 已有 33 字节完整包：采样与解码配对实验', '',
      '三个既有 SF10/BW125k 文件，共 28 个包、24 个不同载荷。原始采集 IQ 保持不变，没有截短载荷，也没有下载新数据。', '',
      '使用干净定位、CFO、LDRO 和帧范围；header 与 payload 均从相应样点解码。CRC 只作最终验收，载荷逐字节匹配干净参考才计为成功。不是连续流接收实验。', '',
      '每个加噪点为 28 包 × 3 个新种子，所有方法配对使用同一 BW 网格噪声。RGR 是采集信号功率与新增噪声功率之比，包含原有噪声；不是标定后的 RF SNR。', '',
      '| 每符号样点与解码 | 原始 IQ | −10 dB | −12 dB | −14 dB | −16 dB |',
      '|---|---:|---:|---:|---:|---:|']
names={'hard':'硬判决','soft':'软 Hamming','joint':'块联合 Top-2'}
for pattern in ('uniform1024','uniform512','random512','uniform256','random256'):
    for method in ('hard','soft','joint'):
        cells=[lookup[(level,pattern,method)] for level in ('clean','-10','-12','-14','-16')]
        text.append('| '+pattern+' / '+names[method]+' | '+' | '.join(r['exact']+'/'+r['trials'] for r in cells)+' |')
text += ['', '计算成本只统计缓存模板之后的选点解调和解码，包含最终 codec，不含 IO、同步和初始化。时间是单轮 Python 诊断，不是优化实现的速度上限。失败可以提前退出，因此下面只比较原始 IQ 下成功的处理时间。', '',
         '| 配置 | 原始 IQ 成功数 | 成功处理时间中位数 / ms | 显式字典大小 / MiB |',
         '|---|---:|---:|---:|']
for pattern in ('uniform1024','uniform512','random512','random256'):
    for method in ('soft','joint'):
        r=lookup[('clean',pattern,method)]
        ms=f"{float(r['median_success_ms']):.3f}" if r['median_success_ms'] else '—'
        text.append(f"| {pattern} / {names[method]} | {r['exact']}/{r['trials']} | {ms} | {int(r['dictionary_bytes'])/2**20:.1f} |")
text += ['', 'FFT 路径没有本实验的显式匹配字典，表中的零不表示总内存为零。随机字典使用 complex128；可优化，不能将当前开销当成随机采样必然下界。', '',
         '均匀 256 点的块联合版本遇到秩亏就拒绝；它的零成功率是这个解码器的结果，不能推广成所有接收方法的普遍不可能性。软译码为项目现有位级 Hamming 实现，联合译码为有限 Top-2 块列表，均非穷举 ML，也不是完整复现某篇最新论文。', '',
         '随机采样每个预算仅用一组预先固定的嵌套位置，仍需检查采样位置变化及独立采集泛化。误接受数只针对含真实信号的本轮试验，不代表纯噪声虚警率。', '',
         '配置、噪声种子、采样位置、逐包状态、计时和源码快照位于本目录。']
text += ['', '**本轮判断。** 均匀半速采样的块联合解码确实利用了位级软译码丢失的候选相关性，在 −12 dB 为 80/84；同预算随机采样加软译码为 81/84。但到 −14 dB，二者为 46/84 和 70/84。现有 Top-2 联合算法不是普遍优于软译码的接收器。', '',
         '同一块联合解码器下，原始 IQ 的均匀 512 点路径约 3.57 ms，随机 512 点路径约 7.57 ms；前者不需要后者的 8 MiB 显式模板字典。这是当前 Python 实现的资源取舍，尚不能称为优化实现的普遍优势。', '',
         '后续优先解决均匀半速路径在较低 RGR 条件下的候选列表截断损失，并与软信息基线比较；保留 1024 点强基线和 256 点失败配置。不能只挑 −12 dB 附近的点证明“无损减半”。', '',
         f"本轮共记录 {sum(int(r['false_accept']) for r in rows)} 条 CRC 通过但载荷不匹配的结果，均未计入成功。逐例复核见 false_accept_audit.json；不同方法可能对应同一个带噪观测，不能视为独立错误事件。"]
(args.directory/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
xs=[-16,-14,-12,-10]
for ax,patterns,title in zip(axes,[('uniform1024','uniform512'),('random512','random256')],['Uniform sampling','Random sampling']):
    for pattern in patterns:
        for method,style in [('soft','--'),('joint','-')]:
            ys=[int(lookup[(str(x),pattern,method)]['exact'])/84 for x in xs]
            ax.plot(xs,ys,style,marker='o',label=f'{pattern}: {method}')
    ax.set_title(title); ax.set_xlabel('Added-noise RGR (dB)'); ax.set_xticks(xs)
    ax.set_ylim(-.03,1.03); ax.grid(alpha=.25); ax.legend(fontsize=8)
axes[0].set_ylabel('Exact frame recovery fraction')
fig.suptitle('Existing SF10 / 33-byte OTA frames; clean timing and CFO supplied')
fig.text(.5,.01,'28 captured packets × 3 paired noise seeds per point; not 84 independent OTA packets',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.94))
fig.savefig(args.directory/'recovery.png',dpi=180)
fig.savefig(args.directory/'recovery.svg')
print(args.directory/'RESULTS.md')
