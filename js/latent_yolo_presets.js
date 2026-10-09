// 千绘 · Latent智能修复 —— 配置预设联动
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

    // 被预设覆盖的控件名 = 各档预设表键的并集（表到位前为空集）。
    function coveredKeys() {
        const seen = {};
        const tables = (presetData && presetData.presets) || {};
        for (const tier of Object.keys(tables)) {
            for (const k of Object.keys(tables[tier])) seen[k] = true;
        }
        return Object.keys(seen);
    }

    // 用户手动改了被覆盖的控件 → 把下拉框切到 自定义。
    // 后端每次运行都会重按预设覆盖这 19 个控件（prepare.py 先读完 widget 再 apply_preset），
    // 所以不切的话用户刚改的值下一次运行就被丢掉；而 自定义 在 presets.py 里 early-return。
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
        const table = (presetData.presets || {})[name];
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
            if (nodeData.name !== "QianhuiLatentYOLO") return;

            // 挂「被覆盖控件 → 翻自定义」监听 + 补应用表到达前的选择。
            // 两件事都依赖拉取到的表，故 onNodeCreated 与 fetch 回调两个时机都调它
            // （谁先到都行）；_qhPresetCovered 保证只包装一次。
            nodeType.prototype._qhPresetSync = function () {
                if (!presetData) return;
                const node = this;
                if (!this._qhPresetCovered) {
                    this._qhPresetCovered = true;
                    for (const name of coveredKeys()) {
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
                    fetch("/qianhui_preset_data")
                        .then(function (r) { return r.json(); })
                        .then(function (d) {
                            presetData = d;
                            // 表晚于节点到达（首屏必然如此）→ 在这里补挂 + 补应用
                            if (node._qhPresetSync) node._qhPresetSync();
                        })
                        .catch(function () { });   // 拉取失败：后端兜底仍生效
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
