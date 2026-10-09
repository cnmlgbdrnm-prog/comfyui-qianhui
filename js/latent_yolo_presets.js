// 千绘 · Latent智能修复 —— 配置预设联动（T12：按节点分表）
// 兼容写法：老前端提供全局 app；新前端(>=0.31) 以 ES module 执行扩展，
// 裸 app 标识符会 ReferenceError，需动态 import 兜底。
(async function () {
    "use strict";
    const WIDGET = "配置预设";
    const CUSTOM = "自定义";          // 与后端 presets.CUSTOM 同字面量
    let presetData = null;
    // 预设程序化写入控件期间为 true。applyToNode 写值后会触发每个被写控件的 callback，
    // 而那个 callback 正是下面挂的「用户改动 → 翻自定义」监听器——不抑制的话，
    // 选中预设的瞬间下拉框就把自己翻成 自定义，预设当场自我否决。
    let applying = false;

    // 4 个节点的注册名 → 节点短名。与后端 presets.NODE_OF 同构。
    //
    // ★ 这张表**同时是 beforeRegisterNodeDef 的闸门**，所以必须是静态的：
    //   那一刻 /qianhui_preset_data **还没回来**（presetData 恒为 null），
    //   拿 payload 决定「要不要给这个节点挂东西」会让整条联动永不注册 ——
    //   症状是「预设面板在，选了什么都不动」，且不报错。
    //   payload 回来后下面 fetchPresetData 会与这张表**逐键比对**，
    //   对不上就整体停用（宁可不动，也不要按一张错的表去写用户的控件）。
    const NODE_OF = {
        QianhuiYoloDetect: "detect",
        QianhuiSamRefine: "sam",
        QianhuiRegionRepair: "repair",
        QianhuiGlobalPass: "global",
    };

    // ★ 临时兼容：巨型节点 QianhuiLatentYOLO（单体）**还在注册表里**。
    //   它不在 NODE_OF 里，但它的「配置预设」下拉必须照旧能用 —— 否则这一轮就是
    //   「4 个新节点能用、老节点坏了」的半开状态，正是本计划一直在防的那种。
    //   所以给单体合成一张**各节点表的并集**，等于 T12 之前的行为。
    //   ★ 单体退役（T13）时，这个常量与下面 buildMonolithTables 的调用一起删掉。
    const MONOLITH = "QianhuiLatentYOLO";

    function buildMonolithTables(presets) {
        const nodeKeys = Object.keys(presets);
        if (!nodeKeys.length) return {};
        const tiers = Object.keys(presets[nodeKeys[0]] || {});
        const out = {};
        for (const tier of tiers) {
            const acc = {};
            for (const nk of nodeKeys) Object.assign(acc, presets[nk][tier] || {});
            out[tier] = acc;
        }
        return out;
    }

    async function getApp() {
        if (typeof app !== "undefined") return app;
        if (globalThis.app) return globalThis.app;
        try {
            const mod = await import("../../scripts/app.js");
            return mod.app || globalThis.app || null;
        } catch (e) {
            return null;
        }
    }

    // 拉表。两件事都在这里做：存表 + 与静态 NODE_OF 对账（对不上就停用）。
    // ★ 返回 Promise，让调用方在需要时能等它；onNodeCreated 那条路不 await
    //   （与 T12 之前一样的时序），只是把逻辑收进一个函数，行为不变。
    let presetFetch = null;
    function fetchPresetData(node) {
        if (!presetFetch) {
            presetFetch = fetch("/qianhui_preset_data")
                .then(function (r) { return r.json(); })
                .then(function (d) {
                    const remote = (d && d.nodes) || {};
                    for (const k of Object.keys(NODE_OF)) {
                        if (remote[k] !== NODE_OF[k]) {
                            // 前后端对不上：可能是版本不一致（后端还没升级）。
                            // 这时按表写值有写错控件的风险，直接**停用前端联动**；
                            // 后端 parse_config → apply_preset 照常兜底。
                            console.warn(
                                "[qianhui] 预设表与前端节点表不一致（" + k + "），"
                                + "已停用预设联动，请更新千绘插件");
                            return;
                        }
                    }
                    // 单体的并集表（见 MONOLITH 的注释；T13 退役时一并删掉）
                    const all = (d && d.presets) || {};
                    if (all && Object.keys(all).length) {
                        d.presets = Object.assign({}, all);
                        d.presets[MONOLITH] = buildMonolithTables(all);
                    }
                    presetData = d;
                })
                .catch(function () { });   // 拉取失败：后端兜底仍生效
        }
        return presetFetch.then(function () {
            if (node && node._qhPresetSync) node._qhPresetSync();
        });
    }

    // 本节点被预设覆盖的控件名 = **本节点那一张表**各档键的并集（表到位前为空集）。
    // ★ T12 起必须传 nodeKey：不传就是整张并集，会把别的节点的控件也挂上监听。
    function coveredKeys(nodeKey) {
        const seen = {};
        const tables = (presetData && presetData.presets && presetData.presets[nodeKey]) || {};
        for (const tier of Object.keys(tables)) {
            for (const k of Object.keys(tables[tier])) seen[k] = true;
        }
        return Object.keys(seen);
    }

    // 用户手动改了被覆盖的控件 → 把下拉框切到 自定义。
    // 后端每次运行都会重按预设覆盖**本节点**的那些控件（prepare.py 先读完 widget
    // 再 apply_preset(node_key=...)），所以不切的话用户刚改的值下一次运行就被丢掉；
    // 而 自定义 在 presets.py 里 early-return。
    function flipToCustom(node) {
        if (applying) return;                      // 预设自己写进去的，不算用户改动
        const w = (node.widgets || []).find(function (x) { return x.name === WIDGET; });
        if (!w || w.value === CUSTOM) return;
        w.value = CUSTOM;
        // 同步触发它的 callback：与 role_random.js 一致，让别的监听者也看得见这次切换。
        // applyToNode 对 自定义 会走 `if (!table) return`，不会递归、也不会再翻。
        if (typeof w.callback === "function") w.callback(w.value);
    }

    function applyToNode(node, name) {
        if (!presetData) {
            // 表还没到：先记住这次选择，等 /qianhui_preset_data 回来在 _qhPresetSync 里补应用。
            // 不记的话用户「选了预设但面板纹丝不动」，而后端兜底照常生效，
            // 于是运行结果与面板显示不一致。
            node._qhPendingPreset = name;
            return;
        }
        const nodeKey = node._qhNodeKey;
        if (!nodeKey) return;                      // 不是 4 个节点之一：本扩展不管
        const perNode = (presetData.presets || {})[nodeKey] || {};
        const table = perNode[name];
        if (!table) return;
        node._qhPendingPreset = null;
        applying = true;
        try {
            for (const key of Object.keys(table)) {
                const w = node.widgets.find(function (x) { return x.name === key; });
                if (!w) continue;
                w.value = table[key];
                if (typeof w.callback === "function") w.callback(w.value);
            }
        } finally {
            applying = false;            // 某个控件 callback 抛异常也不能把标志卡死
        }
    }

    const appRef = await getApp();
    if (!appRef) return;

    appRef.registerExtension({
        name: "qianhui.latent_yolo_presets",
        async beforeRegisterNodeDef(nodeType, nodeData) {
            // ★ 单体走 MONOLITH 那条临时路径（见上面的注释）；4 个新节点走 NODE_OF。
            const nodeKey = (nodeData.name === MONOLITH) ? MONOLITH : NODE_OF[nodeData.name];
            if (!nodeKey) return;                  // 其余节点一律不管

            // 挂「被覆盖控件 → 翻自定义」监听 + 补应用表到达前的选择。
            // 两件事都依赖拉取到的表，故 onNodeCreated 与 fetch 回调两个时机都调它
            // （谁先到都行）；_qhPresetCovered 保证只包装一次。
            nodeType.prototype._qhNodeKey = nodeKey;
            nodeType.prototype._qhPresetSync = function () {
                if (!presetData) return;
                const node = this;
                if (!this._qhPresetCovered) {
                    this._qhPresetCovered = true;
                    for (const name of coveredKeys(nodeKey)) {
                        if (name === WIDGET) continue;   // 别给下拉框自己挂，会自我否决
                        const w = (this.widgets || []).find(function (x) { return x.name === name; });
                        if (!w) continue;
                        w.callback = (function (prev) {
                            return function (v) {
                                const r = prev ? prev.call(this, v) : undefined;
                                flipToCustom(node);
                                return r;
                            };
                        })(w.callback);
                    }
                }
                if (this._qhPendingPreset) applyToNode(this, this._qhPendingPreset);
            };

            const orig = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const ret = orig ? orig.apply(this, arguments) : undefined;
                const node = this;
                if (!presetData) {
                    // 表晚于节点到达（首屏必然如此）→ 拉回后在 _qhPresetSync 里补挂 + 补应用
                    fetchPresetData(node);
                }
                const w = this.widgets.find(function (x) { return x.name === WIDGET; });
                if (w) {
                    w.callback = (function (prev, nd) {
                        return function (v) {
                            const r = prev ? prev.call(this, v) : undefined;
                            applyToNode(nd, v);
                            return r;
                        };
                    })(w.callback, node);
                }
                this._qhPresetSync();     // 表已到位（第二个及以后的节点）则立刻挂上
                return ret;
            };
        },
    });
})();
