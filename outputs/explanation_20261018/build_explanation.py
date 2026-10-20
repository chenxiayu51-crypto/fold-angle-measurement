"""Reproduce teaching diagrams and audit the existing clock fit; never edit input logs."""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Arc, Circle, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.optimize import least_squares

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT/'outputs/experiment_20261016_110453'
sys.path.insert(0, str(ROOT))
from analyze_imu_log import infer_clock_map

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'], 'axes.unicode_minus': False,
                     'font.size': 12, 'figure.facecolor': '#fbfcff',
                     'axes.facecolor': '#fbfcff', 'savefig.facecolor': '#fbfcff'})
H, E1, E2, ACC, GYRO = '#863caf', '#17805b', '#d17a10', '#2174bc', '#c44444'
P = json.loads((OLD/'csv_to_angle_parameters.json').read_text(encoding='utf-8'))
h = np.array(P['axis_B']); e1 = np.array(P['e1']); e2 = np.array(P['e2'])
a0 = np.array(P['gravity_reference']); p0 = a0-(a0@h)*h
u1 = P['first_three_updates'][1]
a = np.array([u1[f'acc_{q}_m_s2'] for q in 'xyz'])


def save(fig, name):
    fig.savefig(OUT/name, dpi=160, bbox_inches='tight')
    plt.close(fig)


def arrow(ax, start, end, color, label=None, offset=(0, 0), lw=2.7):
    ax.annotate('', xy=end, xytext=start,
                arrowprops={'arrowstyle': '-|>', 'color': color, 'lw': lw})
    if label:
        ax.text(end[0]+offset[0], end[1]+offset[1], label, color=color,
                ha='center', va='center', fontsize=12)


def clock_evidence():
    c = P['clock_this_recording_only']; source = P['source']
    with Path(source).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    host = [(i, int(r['timestamp_ns'])) for i, r in enumerate(rows)
            if r['sensor'] in ('acc1', 'gyro1')]
    idx = np.array([i for i, _ in host]); ts = [t for _, t in host]
    pairs = []
    for i, r in enumerate(rows):
        if r['sensor'] in ('acc1', 'gyro1'):
            continue
        j = np.searchsorted(idx, i); lo = max(0, j-1); hi = min(j, len(ts)-1)
        x = (int(r['timestamp_ns'])-c['sensor_base_ns'])/1e9
        before = (ts[lo]-c['host_base_ns'])/1e9
        after = (ts[hi]-c['host_base_ns'])/1e9
        y = ((ts[lo]-c['host_base_ns'])+(ts[hi]-c['host_base_ns']))/2e9
        width = (ts[hi]-ts[lo])/1e9
        pairs.append(dict(csv_line=i+2, sensor=r['sensor'], timestamp_ns=int(r['timestamp_ns']),
                          before_csv_line=int(idx[lo])+2, after_csv_line=int(idx[hi])+2,
                          x_A_s=x, before_B_s=before, after_B_s=after, midpoint_B_s=y,
                          width_s=width, selected=int(width < .025),
                          clamped_boundary=int(j == 0 or j == len(ts))))
    x = np.array([r['x_A_s'] for r in pairs]); y = np.array([r['midpoint_B_s'] for r in pairs])
    sel = np.array([r['selected'] for r in pairs], dtype=bool)
    fit = least_squares(lambda z: z[0]*x[sel]+z[1]-y[sel], [1., 0.],
                        loss='soft_l1', f_scale=.005)
    assert np.allclose(fit.x, [c['scale'], c['offset_s']], rtol=0, atol=1e-14)
    native = infer_clock_map(source, c['host_base_ns'], c['sensor_base_ns'])
    assert native['scale'] == fit.x[0] and native['offset_s'] == fit.x[1]
    pred = fit.x[0]*x+fit.x[1]; res = y-pred
    for r, yp, er in zip(pairs, pred, res):
        r.update(predicted_B_s=float(yp), residual_y_minus_prediction_ms=float(er*1000))
    with (OUT/'clock_pairs.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
    no_boundary = sel & ~np.array([r['clamped_boundary'] for r in pairs], dtype=bool)
    comparison = least_squares(lambda z: z[0]*x[no_boundary]+z[1]-y[no_boundary],
                               [1., 0.], loss='soft_l1', f_scale=.005)
    report = dict(source=source, data_rows=len(rows), pairs=len(pairs), selected=int(sel.sum()),
                  excluded=int((~sel).sum()), clamped_boundary=sum(r['clamped_boundary'] for r in pairs),
                  fit=native, examples=[r for r in pairs if r['csv_line'] in (2, 3, 10, 11, 14, 15)],
                  diagnostic_only_refit_without_boundary=dict(scale=float(comparison.x[0]), offset_s=float(comparison.x[1])),
                  selected_residual_p05_p50_p95_ms=np.percentile(res[sel]*1000, [5, 50, 95]).tolist())
    (OUT/'clock_fit_details.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4), layout='constrained')
    ax = axs[0]
    ax.set_title('真实 CSV 行 9 → 10 → 11 → 12：按写入顺序配对')
    ax.plot([0, .013], [1, 1], color=ACC, lw=4, marker='o', ms=9)
    ax.scatter([.0065], [1], color=E2, s=100, zorder=4)
    ax.text(0, 1.15, 'B 行 9\n0.000 s', ha='center', color=ACC)
    ax.text(.013, 1.15, 'B 行 12\n0.013 s', ha='center', color=ACC)
    ax.text(.0065, .85, '取中点 y = 0.0065 s\n供 A 行 10、11 使用', ha='center', color=E2)
    ax.text(.0065, .35, 'A 行 10：x = 0.019997812 s\nA 行 11：x = 0.023997656 s\n这是近似配对，不是两条记录同时采样。', ha='center')
    ax.set(xlim=(-.004, .017), ylim=(0, 1.55), xlabel='B 相对时间（秒）', yticks=[])
    ax = axs[1]
    ax.scatter(x[sel], res[sel]*1000, s=3, alpha=.25, color=ACC, rasterized=True)
    ax.axhline(0, color='black', lw=1)
    ax.axhline(5, color=E2, ls='--'); ax.axhline(-5, color=E2, ls='--')
    ax.set(title=f'{sel.sum()} 个配对点：拟合后的残差', xlabel='A 相对时间 x（秒）',
           ylabel='y − (c x + d)（毫秒）')
    ax.text(.03, .04, '虚线 ±5 ms：鲁棒损失的尺度\n不是硬件同步精度或误差保证', transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=.9, edgecolor='none'))
    save(fig, '06_clock_fit.png')
    return report


def phone_hinge():
    fig = plt.figure(figsize=(15, 6.5), layout='constrained')
    ax = fig.add_subplot(121)
    frame = plt.imread(OLD/'frame_0000.png')
    ax.imshow(frame); ax.set(xlim=(380, 920), ylim=(420, 60)); ax.axis('off')
    ax.set_title('真实视频：两条侧边在画面上形成开合夹角')
    ax.plot([540, 831], [310, 313], color='#fff36d', lw=4)
    ax.plot([540, 777], [302, 139], color='#69d6ff', lw=4)
    ax.scatter([540], [305], s=130, color=H, edgecolor='white', zorder=8)
    box = dict(facecolor='white', alpha=.92, edgecolor='none', pad=4)
    ax.text(712, 356, 'A：下半机身（固定）', ha='center', bbox=box)
    ax.text(700, 104, 'B：上半机身（转动）', ha='center', bbox=box)
    ax.annotate('铰链在画面中的位置\nh 沿铰链宽度方向，伸向景深', xy=(540, 305), xytext=(399, 395),
                arrowprops=dict(arrowstyle='->', color=H, lw=2), color=H, fontsize=11, bbox=box)
    ax = fig.add_subplot(122, projection='3d')
    th = np.deg2rad(58); tip = np.array([2.1*np.cos(th), 0, 2.1*np.sin(th)])
    base = [[0, -.7, 0], [2.1, -.7, 0], [2.1, .7, 0], [0, .7, 0]]
    upper = [[0, -.7, 0], [tip[0], -.7, tip[2]], [tip[0], .7, tip[2]], [0, .7, 0]]
    ax.add_collection3d(Poly3DCollection([base], facecolor='#bbc4ce', edgecolor='#44515e', alpha=.8))
    ax.add_collection3d(Poly3DCollection([upper], facecolor='#83c9ed', edgecolor=ACC, alpha=.8))
    ax.plot([0, 0], [-1, 1.2], [0, 0], color=H, lw=5)
    ax.quiver(0, .75, 0, 0, .8, 0, color=H, linewidth=3, arrow_length_ratio=.18)
    ax.text(0, 1.6, 0, 'h：沿铰链的单位方向', color=H)
    tt = np.linspace(0, th, 40)
    ax.plot(.8*np.cos(tt), np.full(40, -.82), .8*np.sin(tt), color=E2, lw=3)
    ax.text(.82, -.96, .28, '开合角 θ', color=E2)
    ax.text(1.8, .15, -.15, 'A 固定'); ax.text(tip[0], .1, tip[2]+.15, 'B 转动', color=ACC)
    ax.text(.05, -.95, 1.8, '运动发生在垂直 h 的平面中', color=H)
    ax.view_init(elev=24, azim=-60); ax.set_box_aspect((2.5, 2.7, 2))
    ax.set(xlim=(-.3, 2.5), ylim=(-1, 1.8), zlim=(-.2, 2.1)); ax.set_axis_off()
    ax.set_title('结构示意：h 不是侧边，也不是交点')
    fig.suptitle('先分清“铰链的位置”与“转轴的方向”', fontsize=19)
    fig.supxlabel('照片未标定芯片安装 XYZ；右图只表示机械结构，不宣称恢复了照片中的传感器安装方向。', fontsize=11)
    save(fig, '01_phone_and_h.png')


def basis_diagram():
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.2), layout='constrained')
    ax = axs[0]; ax.axis('off'); ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.set_title('① 在 B 自身坐标中，把沿铰链的分量去掉')
    lines = [
        ('h = (0, 1, 0)：本次按 B 的 +Y 轴作为转轴', H),
        ('a₀ = (−5.3909, 0.0160206, −8.2248)', ACC),
        ('a₀ · h = 0.0160206', H),
        ('(a₀ · h) h = (0, 0.0160206, 0)', H),
        ('p₀ = a₀ − (a₀ · h) h', '#333333'),
        ('     = (−5.3909, 0, −8.2248)', E1),
        ('只剩 X、Z 分量：这就是垂直转轴的平面', E1),
        ('e₁ = p₀ / |p₀|：只留方向，长度变成 1', E1),
        ('e₂ = −h × e₁：在同一平面中转 90°', E2),
    ]
    for i, (s, color) in enumerate(lines):
        ax.text(.1, 9-i*.95, s, color=color, fontsize=12)
    ax = axs[1]; ax.set_aspect('equal'); ax.set(xlim=(-1.4, 1.4), ylim=(-1.3, 1.2))
    ax.axhline(0, color='#abb5bf', lw=1); ax.axvline(0, color='#abb5bf', lw=1)
    arrow(ax, (0, 0), (1.18, 0), '#555555', '+X', (.1, 0))
    arrow(ax, (0, 0), (0, 1.05), '#555555', '+Z', (.13, 0))
    ax.text(.08, .15, '⊗ h = +Y\n指向纸内', color=H)
    for v, col, label, off in [(e1, E1, 'e₁ = (−0.5482, 0, −0.8364)', (-.10, -.18)),
                               (e2, E2, 'e₂ = (+0.8364, 0, −0.5482)', (0, -.19))]:
        arrow(ax, (0, 0), (v[0], v[2]), col, label, off)
    ax.plot([e1[0]*1.13], [e1[2]*1.13], 'o', color=ACC)
    ax.text(-1.32, -.66, 'p₀ 与 e₁ 同向\np₀ 长约 9.834\ne₁ 长度是 1', color=E1, fontsize=11)
    ax.set_title('② 垂直 h 的 XZ 平面：实际数值方向')
    ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('e₁、e₂ 是测角用的两把“方向尺”，不是手机两条侧边', fontsize=18)
    fig.supxlabel('此图 +X 向右、+Z 向上，所以右手坐标的 +Y 指向纸内；⊗ 表示箭尾。', fontsize=11)
    save(fig, '02_reference_basis.png')


def acc_projection():
    u, v = float(a@e1), float(a@e2)
    planar = a-(a@h)*h
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    ax = axs[0]; ax.set_aspect('equal'); ax.set(xlim=(-7, 5), ylim=(-10, 1.8))
    ax.axhline(0, color='#bfc5cb'); ax.axvline(0, color='#bfc5cb')
    arrow(ax, (0, 0), (4.1, 0), '#555', '+X', (.4, 0))
    arrow(ax, (0, 0), (0, 1.1), '#555', '+Z', (.6, .1))
    arrow(ax, (0, 0), (4*e2[0], 4*e2[2]), E2, 'e₂ 方向', (.45, -.5))
    arrow(ax, (0, 0), (4*e1[0], 4*e1[2]), E1, 'e₁ 方向', (-1, 0))
    ax.plot([0, u*e1[0]], [0, u*e1[2]], color=E1, lw=3, alpha=.55)
    ax.plot([u*e1[0], planar[0]], [u*e1[2], planar[2]], color=E2, lw=5)
    arrow(ax, (0, 0), (planar[0], planar[2]), ACC)
    ax.text(-6.7, -9.4, '当前 a 的平面投影\n(−5.3598, 0, −8.2942)', color=ACC)
    ax.text(.5, -5.0, '⊗ h = +Y\n两把方向尺\n与上一张完全相同', color=H, fontsize=11)
    ax.set_title('① 传感器 XZ 图：当前向量几乎沿 e₁')
    ax.set_xticks([]); ax.set_yticks([])
    ax = axs[1]; ax.set(xlim=(-.45, 11.3), ylim=(-.3, 1.1))
    ax.axhline(0, color='#bfc5cb'); ax.axvline(0, color='#bfc5cb')
    arrow(ax, (0, 0), (10.7, 0), E1)
    arrow(ax, (0, 0), (0, .95), E2)
    arrow(ax, (0, 0), (u, v), ACC)
    ax.plot([u, u], [0, v], color=E2, lw=3)
    ax.text(6.5, -.18, 'u = a · e₁ = 9.875075', color=E1)
    ax.text(.3, .87, 'v = a · e₂', color=E2)
    ax.text(4, .55, 'v = 0.064055\nφ = atan2(v, u)\n   = 0.371644°', color=ACC)
    ax.annotate('真实小夹角', xy=(6, v*6/u), xytext=(6.7, .3),
                arrowprops=dict(arrowstyle='->', color=ACC), color=ACC)
    ax.text(.3, -.27, '横向沿 e₁；纵向沿 e₂（纵轴放大，勿从图形量角）', fontsize=10)
    ax.set_title('② 换成 e₁/e₂ 坐标：同一向量的两个读数')
    ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('投影就是“沿某个已知方向，到底占了多少”', fontsize=18)
    save(fig, '03_acc_projection.png')


def gyro_integration():
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    ax = axs[0]; ax.axis('off'); ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.set_title('① Gyro 是绕各轴的角速度，不是移动速度')
    lines = [('原始 g₁ = (0.0084, 0.0349, −0.0032) rad/s', GYRO),
             ('零偏 b = (0.00345, 0.00020, −0.00085)', '#555'),
             ('校正 gc = g₁ − b', GYRO),
             ('     = (0.00495, 0.03470, −0.00235)', GYRO),
             ('h = (0, 1, 0)', H),
             ('r₁ = gc · h = 0 + 0.03470 + 0', H),
             ('     = 0.03470 rad/s（绕铰链的部分）', H),
             ('本次正向轴速度对应合拢：θ̇ = −r', '#333'),
             ('所以 θ̇₁ = −1.988164 °/s', '#333')]
    for i, (s, color) in enumerate(lines): ax.text(0, 9-i*.95, s, color=color, fontsize=12)
    ax = axs[1]
    t = np.array([0, .013]); r = np.array([.0416, .0347])
    ax.fill_between(t*1000, 0, r, color=GYRO, alpha=.22, label='梯形面积 = 角位移（rad）')
    ax.plot(t*1000, r, 'o-', color=GYRO, lw=3, ms=8)
    ax.plot([0, 13], [.0416, .0416], '--', color='#999', lw=1)
    ax.plot([0, 13], [.0347, .0347], '--', color='#999', lw=1)
    ax.axhline(0, color='#333')
    ax.set(xlim=(-1.2, 14.2), ylim=(-.003, .053), xlabel='相对时间（毫秒）',
           ylabel='沿 h 的角速度 r（rad/s）', title='② 13 ms 内，角度变化是曲线下的面积')
    ax.text(1.1, .013, '(0.0416 + 0.0347) / 2 × 0.013\n= 0.00049595 rad\n开合角增量 = −0.02841584°', fontsize=12)
    ax.annotate('r₀ = 0.0416', xy=(0, r[0]), xytext=(.1, .048), color=GYRO)
    ax.annotate('r₁ = 0.0347', xy=(13, r[1]), xytext=(8.2, .043), color=GYRO)
    ax.legend(loc='lower left', fontsize=10)
    fig.suptitle('先取绕铰链的角速度，再把角速度随时间累积成角度', fontsize=18)
    save(fig, '04_gyro_trapezoid.png')


def observability():
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.2), layout='constrained')
    for ax in axs:
        ax.set(xlim=(-2, 2), ylim=(-1.7, 2)); ax.set_aspect('equal'); ax.axis('off')
    ax = axs[0]
    ax.plot([-.2, 1.5], [-.7, -.7], color='#6c7680', lw=10)
    ax.plot([-.2, .8], [-.7, .95], color=ACC, lw=10)
    ax.add_patch(Circle((-.2, -.7), .12, facecolor='white', edgecolor=H, lw=3))
    ax.text(-.2, -.7, '×', color=H, ha='center', va='center', fontsize=14)
    ax.text(-1.9, -.95, 'h 水平，指向纸内', color=H)
    arrow(ax, (-1.1, -.3), (-1.1, 1.1), ACC, '静止 Acc 向量', (0, .24))
    arrow(ax, (1.5, .7), (1.5, -.4), '#666', '物理重力向下', (-.4, .95))
    ax.text(-1.8, -1.55, 'Acc 大部分落在测角平面里\n转动 B，会改变 X/Z 分量 → 能测开合变化', fontsize=11)
    ax.set_title('① 像桌上开合：重力垂直铰链，信息充足')
    ax = axs[1]
    ax.add_patch(Polygon([(-.5, -.8), (.4, -.6), (.4, 1.2), (-.5, 1)], color='#c1c8cf'))
    ax.add_patch(Polygon([(-.5, -.8), (-1.4, -.3), (-1.4, 1.5), (-.5, 1)], color='#83c9ed'))
    arrow(ax, (-.5, -.95), (-.5, 1.65), H, 'h 竖直', (.6, .02))
    arrow(ax, (.95, -.7), (.95, .9), ACC, '静止 Acc', (0, .3))
    ax.text(-1.8, -1.55, '像门绕竖直轴转：Acc 几乎沿 h\n扣掉轴向分量后接近 0 → 无法靠重力区分角度', fontsize=11)
    ax.set_title('② 把铰链竖起来：Acc 对开合角几乎不敏感')
    fig.suptitle('为什么不是每次 Acc 都能校正？先看重力能否“看见”开合', fontsize=18)
    fig.supxlabel('静止时原始 Acc 通常测支撑产生的比力，与物理重力反向；图中蓝箭头表示静止 Acc，不是向下的重力。', fontsize=10)
    save(fig, '05_acc_observability.png')


def body_and_world():
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.8), layout='constrained')
    # Choose a schematic installation that maps the recorded reference e1 to world-up.
    # This fixes a drawing convention; it is not an optical mounting calibration.
    alpha0 = np.pi/2-np.arctan2(e1[2], e1[0])
    origin = np.array([0., 0.])
    for ax, delta in zip(axs, [0., np.pi/3]):
        alpha = alpha0+delta
        xx = np.array([np.cos(alpha), np.sin(alpha)])
        zz = np.array([-np.sin(alpha), np.cos(alpha)])
        bangle = np.deg2rad(P['initial_angle_deg'])+delta
        tip = 1.9*np.array([np.cos(bangle), np.sin(bangle)])
        ax.plot([0, 2.1], [0, 0], color='#89949f', lw=11, solid_capstyle='round')
        ax.plot([0, tip[0]], [0, tip[1]], color=ACC, lw=11, solid_capstyle='round')
        ax.text(1.9, -.25, 'A 固定', ha='center')
        ax.text(tip[0]+.15, tip[1]+.1, 'B', color=ACC)
        arrow(ax, origin, .94*xx, '#555555', '+X', (.0, -.12))
        arrow(ax, origin, .94*zz, '#555555', '+Z', (.0, -.15))
        v1 = e1[0]*xx+e1[2]*zz
        v2 = e2[0]*xx+e2[2]*zz
        arrow(ax, origin, 1.12*v1, E1, 'e₁', (-.15, .03))
        arrow(ax, origin, 1.12*v2, E2, 'e₂', (-.1, -.1))
        ax.add_patch(Circle((0, 0), .075, facecolor='white', edgecolor=H, lw=2, zorder=10))
        ax.text(0, 0, '×', color=H, ha='center', va='center', zorder=11)
        ax.text(.13, -.35, 'h = +Y 指向纸内', color=H, fontsize=10)
        arrow(ax, (-1.65, .1), (-1.65, 1.65), ACC, '静止 Acc\n世界方向向上', (0, .25))
        arrow(ax, (2.3, 1.65), (2.3, .45), '#777777')
        ax.text(2.3, 1.95, '物理重力\n向下', ha='center', color='#666666', fontsize=11)
        ax.set_aspect('equal'); ax.set(xlim=(-2.3, 2.8), ylim=(-1.6, 2.6)); ax.axis('off')
        title = '初始化：e₁ 对准初始 Acc 的平面方向' if delta == 0 else 'B 再打开 60°：坐标尺随 B 转，重力不跟着转'
        ax.set_title(title, fontsize=13)
        footer = ('XZ 分量：a / |a| ≈ (−0.548, −0.836)\n在 e₁/e₂ 尺上：φ ≈ 0°' if delta == 0 else
                  '在 e₁/e₂ 尺上：φ ≈ −60°\n开合角 θ = θ₀ − φ → 增加 60°')
        ax.text(.1, -1.35, footer, ha='center', fontsize=12)
    fig.suptitle('把手机、h、XYZ、e₁/e₂、静止 Acc 与物理重力放到一起看', fontsize=18)
    fig.supxlabel('安装示意，不是照片标定：选择与初始读数一致的朝向来讲几何；所有坐标箭头平移到铰链处便于比较。', fontsize=11)
    save(fig, '07_phone_body_coordinates.png')


def main():
    report = clock_evidence()
    phone_hinge(); basis_diagram(); acc_projection(); gyro_integration(); observability(); body_and_world()
    assert np.allclose([np.linalg.norm(h), np.linalg.norm(e1), np.linalg.norm(e2)], 1)
    assert np.allclose([h@e1, h@e2, e1@e2], 0, atol=1e-14)
    assert np.allclose(np.cross(e1, e2), -h)
    assert np.allclose(p0/np.linalg.norm(p0), e1)
    assert abs(float(np.arctan2(a@e2, a@e1))-u1['acc_phase_rad']) < 1e-14
    # Orthogonal reconstruction includes the component parallel to the hinge.
    assert np.allclose(a, (a@e1)*e1+(a@e2)*e2+(a@h)*h)
    for path in OUT.glob('*.png'):
        im = plt.imread(path); assert im.shape[0] > 500 and im.shape[1] > 1000
    checks = dict(clock_coefficients_match_existing_code=True, basis_orthonormal=True,
                  e1_cross_e2_equals_negative_h=True, vector_reconstruction=True,
                  actual_acc_phase_verified=True, png_count=len(list(OUT.glob('*.png'))))
    (OUT/'verification.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == '__main__': main()
