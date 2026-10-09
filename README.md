# 千绘 (comfyui-qianhui)

ComfyUI 综合提示词插件，由五个历史插件整合而成：

| 来源插件 | 内容 |
|---|---|
| comfyui-FMC | 智能负面提示词节点 |
| ComfyUIcustom_nodes | 剧情编排机、剧情播放器、角色定制、场景光效大师 |
| confyui-pose | 肢体动作师、姿势大师、战斗指挥官 |
| commfyui-loraxy | 多 LoRA XY 网格测试节点 |
| 水印插件 | Latent YOLO 智能修复（含 SAM/ControlNet）、水印添加、水印提取 |

## 安装

将 `comfyui-qianhui` 整个文件夹放入 ComfyUI 的 `custom_nodes/` 目录，重启 ComfyUI 即可。
节点在右键菜单中的分组统一为 **千绘/**（提示词、剧情、角色、场景、动作、姿势、战斗、网格、修复、水印）。

## 节点列表

| 节点 | 注册名 | 说明 |
|---|---|---|
| 千绘 · 智能负面提示词 | `QianhuiAutoNegative` | 通用负面词 + 正/负面 Embeddings + 负面整体权重 |
| 千绘 · 剧情编排机 | `QianhuiStoryComposer` | 剧情/人设/全局设定/手动补充四路合并、加权、去重 |
| 千绘 · 剧情播放器 | `QianhuiStoryScheduler` | 按剧本 JSON 逐段输出 Tag 与页面引导，支持自动连播/手动页码 |
| 千绘 · 角色定制 | `QianhuiCharacterCreator` | 30 个维度组合角色提示词，素材在 `data/` |
| 千绘 · 场景光效大师 | `QianhuiSceneMaster` | 场景/季节/天气/时间/光效/画质风格/材质数据库；全手动选择，输出纯标签（原权重滑杆已移除），画质风格内置 anima 完整质量体系（官方顶配与美学评分 5 档）与 Krea 2 画风词库（官方 Moodboard 画风 6 款 + 风格 LoRA 触发词 9 个 + Krea2 Me 质量词），氛围句池提供单句环境自然语言（anima Hybrid 第二层），选质量档 + 氛围句即符合官方规范 |
| 千绘 · 肢体动作师 | `QianhuiActionMaster` | 左右手/左右腿独立控制，随机或手动模式；输出纯标签（不加权重），标签下划线统一转空格 |
| 千绘 · 姿势大师 | `QianhuiSexMaster` | 姿势分类数据池，智能随机/手动指定，行为类型自动判定；模型类型切换 SDXL / anima（权重 2 起步且最多 3 个、基础与细节标签去权重，符合 anima 官方规范） |
| 千绘 · 战斗指挥官 | `QianhuiCombatMaster` | 武器/招式/战损/特效/运镜组合，随机激战或手动指令；武器-招式组联动防错配；输出纯标签（不加权重） |
| 千绘 · 角色随机师 | `QianhuiRandomCharacter` | V2：分类筛选（不限/女/男/兽类·福瑞/机械·人造人/兽娘/机械娘）+ 手动指定（选作品→角色下拉联动→指定角色或作品内随机）；Danbooru 高频角色两级随机，按模型自动转义；1629 角色分类标签与全量汉化（英文口输出供提示词使用，中文口输出显示名） |
| 千绘 · 多LoRA XY网格 | `QianhuiMultiLoRAGrid` | XY 网格批量对比 LoRA（行=最多 6 个 LoRA + No LoRA 基线行，列=X 轴统一权重），逐格挂载采样、实时预览、拼图输出；自动溯源底模名称与正/负向提示词绘制在信息栏 |
| 千绘 · Latent智能修复 | `QianhuiLatentYOLO` | YOLO 多尺度检测 + Tile 分块 + Seg 分割 + SAM 精修 + ControlNet + 手脚增强；`配置预设` 五档一键调速/调质 |
| 千绘 · 水印添加 | `QianhuiWatermarkAdd` | 可见水印（文字/图片平铺）与 DCT-QIM 隐形水印；依赖 `reedsolo` |
| 千绘 · 水印提取 | `QianhuiWatermarkExtract` | 提取 DCT-QIM 隐形水印文本 |

## 目录结构

```
comfyui-qianhui/
├── __init__.py          # 注册入口 (13 节点统一 Qianhui 前缀)
├── nodes/               # 节点实现
│   ├── negative_node.py     # 智能负面提示词
│   ├── story_node.py        # 剧情播放器 + 剧情编排机
│   ├── character_node.py    # 角色定制
│   ├── scene_node.py        # 场景光效大师
│   ├── pose_node.py         # 肢体动作师
│   ├── sex_pose_node.py     # 姿势大师
│   ├── combat_node.py       # 战斗指挥官
│   ├── role_name_node.py    # 角色随机师
│   ├── lora_grid_node.py    # 多LoRA XY网格
│   ├── qianhui_latent_yolo/ # Latent 智能修复（支撑模块 + 6 Mixin + 5 阶段函数）
│   └── qianhui_watermark/   # 水印添加 / 提取
├── data/                # 角色素材 + 高频角色名库(role_names/role_owner/role_names_cn)
├── story_scripts/       # 剧本文件
├── pose_data/           # 动作素材
├── configs/             # 姿势分类与配套动作
├── combat_data/         # 战斗素材
└── requirements.txt     # Python 依赖（reedsolo）
```

## 注册约定

- 全部节点注册名统一带 `Qianhui` 前缀，不保留任何旧注册名别名；
  历史 workflow 若引用旧类名（`Gemini_*`、`GeminiSexMasterV39` 等）需在节点图中重新选择新节点。
- 节点 class 名、注册 key 与显示名三者解耦：显示名可随时调整，不影响 workflow。

## 数据目录

| 目录 | 用途 | 格式 |
|---|---|---|
| `data/` | 角色素材（身份/年龄/种族/体型/发型/服装/饰品）+ 高频角色名库（`role_names.json`/`role_owner.json`/`role_names_cn.json`，角色名已全量手工汉化，缺失时回退英文） | 每文件一个 JSON 对象，键对应节点下拉选项 |
| `story_scripts/` | 剧本文件 | JSON 数组，每项 `{"tags": "...", "guide": "..."}` |
| `pose_data/` | 动作素材（视角/全身/头/左右手/左右腿/表情） | JSON 数组 |
| `configs/` | 姿势分类与配套动作 | JSON 对象，键为数据池名 |
| `combat_data/` | 战斗素材（武器/招式/战损/特效/运镜） | JSON 数组 |

所有数据目录均支持热更新：修改 JSON 后刷新 ComfyUI 前端即重新加载。

## 剧本文件格式

`story_scripts/` 下的每个 JSON 文件为一个剧本，格式为数组，每页一项：

```json
[
  {"tags": "page1 tags here, separated by commas", "guide": "该页的分镜引导文字"},
  {"tags": "page2 tags", "guide": "第二页引导"}
]
```

- 选择剧本后，剧情播放器按自动连播（内部计数）或手动锁定（指定页码）输出当前页。
- 切换剧本文件时计数器自动归零；勾选"重置进度"可手动归零。
- 也可不选文件，直接在"临时手写剧本"中按行写入，每行一页。

## 注意事项

- 剧本、姿势等数据文件格式错误时节点会跳过该文件并在控制台打印警告，不影响其他功能。
- 剧情播放器输出"当前段落(Tag)"为纯 Tag 文本，建议经剧情编排机或直接接入正向提示词。
- 4 个提示词节点（角色定制/肢体动作师/场景光效大师/战斗指挥官）的控件标签已去掉数字前缀，
  并移除了 9 个控件（角色定制的「构图范围/Lora触发词/手动补充」、肢体动作师的「手动补充/模型类型」、
  场景光效大师的「手动补充/环境总权重」、战斗指挥官的「模型类型/补充指令」）。节点注册名未变，但旧 workflow 的控件值可能因控件减少而错位，
  建议重新检查一遍参数。
- 肢体动作师、战斗指挥官、场景光效大师改为**纯标签输出**（不含 `(tag:1.2)` 权重）。
  需要加权重时，在后续的提示词节点里自行书写。
- `Latent 智能修复` 与 `水印` 模块需要额外依赖：`pip install reedsolo`（隐形水印的 Reed-Solomon 纠错）。未安装时可见水印仍可用，隐形水印会报出带安装指引的错误。
- 整合自水印插件的 3 个节点**不保留旧注册名**（`LatentKSamplerUpscalerYOLO`、`WatermarkAdd`、`WatermarkExtract` 及中文别名），旧 workflow 需重新选择节点。
