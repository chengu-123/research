## 0. 任务定义
输入只有一张闭合态、固定单视角的铰接物体图像 state0。
最终希望输出：
- 一个 canonical、part-consistent 的 3D 表示
- 将物体分成 base 与 move 两部分
- 恢复单关节运动学参数
- revolute：旋转轴方向、轴上一点、各状态角度
- prismatic：平移方向、各状态位移
- 在 canonical 空间中尽可能补全隐藏几何与隐藏纹理
- 最终可导出 URDF

## 1. 当前核心问题
### P1. 多状态 occupancy 不一致
同一物体的多状态图像输入 TRELLIS / FreeArt3D-TRELLIS 后，会出现：
- 跨状态整体位置偏移
- 表面局部抖动
- 同一物理 base 区域在不同状态下 occupancy 不一致
#### 注意：
这里的目标不是让所有状态完全一样。
真正目标是：
- base 的 occupancy 在多状态下尽可能一致
- move 的 occupancy 仍然保留真实的 state-specific 几何变化
也就是说，要做的是 base-consistent but move-preserving occupancy consistency，而不是把所有状态抹平。

### P2. base–move 几何冲突
#### P2.1 Containment / swept-volume conflict
- 对于抽屉等 prismatic 物体，闭合态时 move_part 被 base/housing 包裹。
- 当把 move_part 逆变换回 canonical 后，可能与 base 的内部 occupancy 冲突。
#### 这类问题的本质是：
- canonical mover 占据的空间
- base/housing 在闭合态的占据空间
- 二者在 voxel 层面不自洽
### P2.2 Boundary surface collision
已知存在问题：
- 在轨迹正确，部件voxel分割正确后，当move_voxel逆变换后会与base重叠（表面层和内部都会）
- 因此如果是仅用一份voxel_occupancy，不能简单二分（分base/move）
而出现表面穿模或边界冲突。

结论：
“把 base 抽走一点”不是统一解法。
P2.1 和 P2.2 必须分别处理。

### P3. 单关节轨迹不够精确
当前难点不是只判断 revolute / prismatic，而是要细粒度优化关节参数。
对于 revolute，真正困难在于：
- 旋转轴方向可能偏
- 旋转轴位置可能偏
- 轴可能漂浮在空间中，不满足真实接触关系
#### 但铰接物体有一个重要物理先验：
- base 和 move 必然存在连接或接触关系
- base–move 边界应形成一个连通或近连通的接触区
- 旋转轴不应任意漂浮，而应穿过该接触边界附近的离散 anchor set

注意：
不能写成“旋转轴绑定某一个 voxel”。
正确表达是：

旋转轴必须穿过 base–move 接触边界附近的离散 anchor set 或其窄邻域。

### P4. 多状态纹理信息没有被充分利用

现有很多方案：
- 要么依赖预训练微调
- 要么只使用单状态纹理
- 要么没有充分利用不同状态对隐藏表面的逐步暴露

但对铰接物体而言：
- state0 能看到外表面
- 打开后的状态能逐步暴露内部纹理、隐藏表面、侧边与背面
- 不同状态能提供不同的可见纹理证据

因此，纹理问题的核心不是“单状态重建纹理”，而是：

如何把不同状态下可见的纹理证据，按置信度与可见性回灌到 canonical 空间，形成统一表示。

### P5. 逐状态补充可见形状未被利用
在state0的时候无法看到（平移/旋转的空间内部形状信息，旋转的move部件的背面形状信息）
在逐步打开的状态下依次显现，内部形状信息和move部件之前不可见部件的形状等


## 2. 当前目标
### G1. 实现多状态 base-consistent occupancy

目标是生成多状态下的 voxel_occupancy，满足：
- base 尽量一致
- move 保持真实状态差异
- 不允许为了“一致性”把 move evidence 一起抹平

这一步的重点不是“视觉上看起来类似”，而是：
- 同一物理 base 区域在 voxel 空间里尽量对齐
- 同一物理 move 区域在不同状态下能够被后续单关节模型解释
### G2. 解决 base–move 几何冲突 （最后处理，先解决其他三个）

这一点目前还没有完全定稿，必须诚实写成 open problem / partially solved problem。

当前理解是：
- 对 P2.1 containment conflict，需要研究
- swept-volume / cavity carving / base editing
- 对 P2.2 boundary collision，需要研究
- contact-band / surface-aware refinement / boundary trimming

这里不能写成“已经解决”。
最多只能写成：

当前已经明确问题分类和物理约束，但具体几何解法仍需进一步验证。

### G3. 优化单关节运动

要做的不是粗略估计轨迹，而是：
- 识别 base 与 move
- 拟合 revolute / prismatic
- 用接触边界先验细粒度优化 joint 参数

具体目标：
对 revolute：
- 恢复轴方向
- 恢复轴位置
- 保证轴穿过接触边界附近 anchor set
对 prismatic：
- 恢复平移方向
- 恢复位移大小
- 保证运动方向与几何结构相容

这个模块的核心物理先验是：

base 与 move 之间应存在一个连通或近连通的接触区；
关节参数必须与该接触区几何兼容。
同一个部件共享一个运动参数

### G4. 做 visibility-aware 的跨状态voxel纹理补全

这里不能简单写成“类似 MorphAny3D 做纹理融合”。
必须写清楚：我们借的是跨状态voxel纹理互补思想，不是直接照搬 morphing 设定。

正确目标是：

- 在正确的 posed/canonical support 上逐状态提取纹理/特征和
- 在正确的 posed/canonical support 上逐状态提取正确的新增/减少voxel（在打开后，柜子内部会呈现真正voxel形状）
- 将不同状态中真正可见的隐藏表面信息逐步回灌到 canonical
- 用可见性和置信度控制 donor 优先级

必须区分三类区域：

#### G4.1 state0 已可见表面

这些区域以 state0 为主锚。
因为它们在闭合态真实可见，置信度最高。

#### G4.2 only-open 才暴露的隐藏表面

这些区域由对应 open states 提供主要 donor 信息。
例如：
- 抽屉内部
- 柜门内侧
- 运动后暴露出的侧面和背面
#### G4.3 所有状态都未真正可见的区域

这些区域不能假装有高置信真纹理。
必须：
- 允许由生成先验补全
- 明确标记为低置信或 uncertain
- 不能与真实可见纹理混为一谈

因此，这里的正确表述应当是：

做 visibility-aware / provenance-aware / confidence-aware 的跨状态纹理补全，
而不是简单地对所有状态纹理做平均。