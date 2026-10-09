// 千绘 · Latent智能修复 —— 配置预设联动
// 兼容写法：老前端提供全局 app；新前端(>=0.31) 以 ES module 执行扩展，
// 裸 app 标识符会 ReferenceError，需动态 import 兜底。
(async function () {
    "use strict";
    const WIDGET = "配置预设";
    let presetData = null;

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

    function applyToNode(node, name) {
        if (!presetData) return;
        const table = (presetData.presets || {})[name];
        if (!table) return;
        for (const key of Object.keys(table)) {
            const w = node.widgets.find(function (x) { return x.name === key; });
            if (!w) continue;
            w.value = table[key];
            if (typeof w.callback === "function") w.callback(w.value);
        }
    }

    const appRef = await getApp();
    if (!appRef) return;

    appRef.registerExtension({
        name: "qianhui.latent_yolo_presets",
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name !== "QianhuiLatentYOLO") return;
            const orig = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const ret = orig ? orig.apply(this, arguments) : undefined;
                const node = this;
                if (!presetData) {
                    fetch("/qianhui_preset_data")
                        .then(function (r) { return r.json(); })
                        .then(function (d) { presetData = d; })
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
                return ret;
            };
        },
    });
})();
