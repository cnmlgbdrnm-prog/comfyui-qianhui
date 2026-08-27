// 千绘 · 角色随机师 —— 作品→角色下拉联动
(function () {
    "use strict";
    const SENTINEL = "（未指定）";
    let roleData = null;

    function roleLabel(entry) {
        const [name, cn] = entry;
        return cn ? cn + " (" + name + ")" : name;
    }

    app.registerExtension({
        name: "qianhui.role_random",
        async beforeRegisterNodeDef(nodeType, nodeData, app) {
            if (nodeData.name !== "QianhuiRandomCharacter") return;
            const orig = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const ret = orig ? orig.apply(this, arguments) : undefined;
                const node = this;
                if (!roleData) {
                    fetch("/qianhui_role_data")
                        .then(function (r) { return r.json(); })
                        .then(function (d) {
                            roleData = d;
                            if (node._syncRoleList) node._syncRoleList();
                        })
                        .catch(function () { }); // 拉取失败: 保持后端全量静态列表
                }
                const workWidget = this.widgets.find(function (w) { return w.name === "指定作品"; });
                const roleWidget = this.widgets.find(function (w) { return w.name === "指定角色"; });
                if (workWidget && roleWidget) {
                    // 作品变化 -> 同步角色列表（callback 内 this 是 widget, 故用闭包 node）
                    workWidget.callback = (function (prev, nd) {
                        return function (v) {
                            const r = prev ? prev.call(this, v) : undefined;
                            if (nd._syncRoleList) nd._syncRoleList(v);
                            return r;
                        };
                    })(workWidget.callback, node);
                }
                return ret;
            };
            nodeType.prototype._syncRoleList = function (workValue) {
                if (!roleData) return;
                const workWidget = this.widgets.find(function (w) { return w.name === "指定作品"; });
                const roleWidget = this.widgets.find(function (w) { return w.name === "指定角色"; });
                if (!workWidget || !roleWidget) return;
                const cur = workValue !== undefined ? workValue : workWidget.value;
                const byWork = roleData.roles_by_work || {};
                let entries = [];
                let tag = null;
                if (cur && cur !== SENTINEL) {
                    const i = cur.lastIndexOf(" (");
                    tag = i === -1 ? cur : cur.slice(i + 2).replace(/\)$/, "").trim();
                }
                if (tag && byWork[tag]) {
                    entries = byWork[tag];
                } else {
                    for (const w of roleData.works || []) {
                        if (byWork[w[0]]) for (const e of byWork[w[0]]) entries.push(e);
                    }
                }
                roleWidget.options.values = [SENTINEL].concat(entries.map(roleLabel));
                if (roleWidget.value !== SENTINEL ||
                    !roleWidget.options.values.includes(roleWidget.value)) {
                    roleWidget.value = SENTINEL;
                    if (typeof roleWidget.callback === "function") roleWidget.callback(roleWidget.value);
                }
            };
        },
    });
})();
