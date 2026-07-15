// UpFound frontend glue — connects the static forms to the backend API.
// Same-origin (served by FastAPI), so paths are relative. Token in localStorage.
(function () {
  "use strict";

  const TOKEN_KEY = "uf_token";
  const getToken = () => localStorage.getItem(TOKEN_KEY);
  const saveToken = (t) => localStorage.setItem(TOKEN_KEY, t);
  const clearToken = () => localStorage.removeItem(TOKEN_KEY);
  window.upfoundLogout = () => { clearToken(); location.href = "login.html"; };

  async function api(path, { method = "GET", body = null, auth = false, form = false } = {}) {
    const headers = {};
    if (auth && getToken()) headers["Authorization"] = "Bearer " + getToken();
    let payload = body;
    if (body && !form) { headers["Content-Type"] = "application/json"; payload = JSON.stringify(body); }
    const res = await fetch(path, { method, headers, body: payload });
    let data = {};
    try { data = await res.json(); } catch (e) { /* non-json */ }
    if (!res.ok) throw new Error(data.detail || ("เกิดข้อผิดพลาด (" + res.status + ")"));
    return data;
  }

  // small toast/inline message helper — inserts a bootstrap alert above a form
  function flash(formEl, message, kind) {
    let box = formEl.querySelector(".uf-flash");
    if (!box) {
      box = document.createElement("div");
      box.className = "uf-flash mt-3";
      formEl.appendChild(box);
    }
    box.innerHTML = '<div class="alert alert-' + (kind || "info") + ' mb-0">' + message + "</div>";
  }

  document.addEventListener("DOMContentLoaded", function () {
    const page = location.pathname.toLowerCase();

    updateNavbar();

    // ---- login.html ------------------------------------------------------
    if (page.endsWith("/login.html") || page.endsWith("login.html")) {
      const form = document.querySelector("form");
      form.addEventListener("submit", async function (e) {
        e.preventDefault();
        try {
          const data = await api("/api/login", {
            method: "POST",
            body: {
              email: document.getElementById("UserEmail").value,
              password: document.getElementById("UserPassword").value,
            },
          });
          saveToken(data.token);
          flash(form, "เข้าสู่ระบบสำเร็จ กำลังพาไป...", "success");
          location.href = "index.html";
        } catch (err) { flash(form, err.message, "danger"); }
      });

      // demo account: one-click sign-in + shown credentials
      api("/api/demo-account").then(function (d) {
        if (!d || !d.enabled) return;
        const box = document.createElement("div");
        box.className = "mt-3 text-center";
        box.innerHTML =
          '<button type="button" class="btn btn-outline-primary w-100" id="uf-demo-btn">เข้าสู่ระบบด้วยบัญชีทดลอง</button>' +
          '<div class="small text-muted mt-2">บัญชีทดลอง — อีเมล: <b>' + d.email + "</b> · รหัส: <b>" + d.password + "</b></div>";
        form.appendChild(box);
        document.getElementById("uf-demo-btn").addEventListener("click", async function () {
          document.getElementById("UserEmail").value = d.email;
          document.getElementById("UserPassword").value = d.password;
          try {
            const data = await api("/api/login", { method: "POST", body: { email: d.email, password: d.password } });
            saveToken(data.token);
            location.href = "index.html";
          } catch (err) { flash(form, err.message, "danger"); }
        });
      }).catch(function () {});
    }

    // ---- Register.html ---------------------------------------------------
    if (page.endsWith("register.html")) {
      const form = document.querySelector("form");
      form.addEventListener("submit", async function (e) {
        e.preventDefault();
        try {
          const data = await api("/api/register", {
            method: "POST",
            body: {
              name: (document.getElementById("UserName") || {}).value || "",
              email: document.getElementById("UserEmail").value,
              password: document.getElementById("UserPassword").value,
            },
          });
          saveToken(data.token);
          flash(form, "สมัครสมาชิกสำเร็จ กำลังพาไป...", "success");
          location.href = "index.html";
        } catch (err) { flash(form, err.message, "danger"); }
      });
    }

    // ---- formitem.html (report a lost item + show matches) ---------------
    if (page.endsWith("formitem.html")) {
      if (!getToken()) { location.href = "login.html"; return; }
      const form = document.querySelector("form");
      form.addEventListener("submit", async function (e) {
        e.preventDefault();
        const fd = new FormData();
        fd.append("itemName", document.getElementById("itemName").value);
        fd.append("itemColor", document.getElementById("itemColor").value);
        fd.append("itemQty", document.getElementById("itemQty").value || "1");
        fd.append("itemType", document.getElementById("itemType").value);
        fd.append("itemLocation", document.getElementById("itemLocation").value);
        fd.append("itemDate", document.getElementById("itemDate").value);
        fd.append("itemDetail", document.getElementById("itemDetail").value);
        const files = document.getElementById("itemImages").files;
        for (let i = 0; i < files.length; i++) fd.append("images", files[i]);

        flash(form, "กำลังค้นหาสิ่งของที่ตรงกัน...", "info");
        try {
          const data = await api("/api/reports", { method: "POST", body: fd, auth: true, form: true });
          renderMatches(form, data);
        } catch (err) {
          if (err.message.indexOf("token") >= 0 || err.message.indexOf("401") >= 0) {
            location.href = "login.html"; return;
          }
          flash(form, err.message, "danger");
        }
      });
    }

    // ---- formperson.html (report a lost person) — public, no login gate --
    if (page.endsWith("formperson.html")) {
      const form = document.querySelector("form");
      form.addEventListener("submit", async function (e) {
        e.preventDefault();
        const fd = new FormData();
        fd.append("personName", document.getElementById("personName").value);
        fd.append("personGender", document.getElementById("personGender").value);
        fd.append("personAge", document.getElementById("personAge").value || "0");
        fd.append("personHeight", document.getElementById("personHeight").value || "0");
        fd.append("personLocation", document.getElementById("personLocation").value);
        fd.append("personDate", document.getElementById("personDate").value);
        fd.append("personDetail", document.getElementById("personDetail").value);
        const files = document.getElementById("personImages").files;
        for (let i = 0; i < files.length; i++) fd.append("images", files[i]);

        flash(form, "กำลังค้นหาบุคคลที่ตรงกัน...", "info");
        try {
          const data = await api("/api/person-reports", { method: "POST", body: fd, auth: true, form: true });
          renderPersonMatches(form, data.person_report_id, data.matches, "พบเห็น");
        } catch (err) { flash(form, err.message, "danger"); }
      });
    }

    // ---- foundperson.html (report a FOUND person + match to lost) --------
    if (page.endsWith("foundperson.html")) {
      const form = document.querySelector("form");
      form.addEventListener("submit", async function (e) {
        e.preventDefault();
        const fd = new FormData();
        fd.append("foundLocation", document.getElementById("foundLocation").value);
        fd.append("foundDate", document.getElementById("foundDate").value);
        fd.append("foundDetail", document.getElementById("foundDetail").value);
        fd.append("foundContact", document.getElementById("foundContact").value);
        const files = document.getElementById("foundImages").files;
        for (let i = 0; i < files.length; i++) fd.append("images", files[i]);

        flash(form, "กำลังค้นหารายงานคนหายที่ตรงกัน...", "info");
        try {
          const data = await api("/api/found-persons", { method: "POST", body: fd, auth: true, form: true });
          renderPersonMatches(form, data.found_report_id, data.matches, "รายงานคนหาย");
        } catch (err) { flash(form, err.message, "danger"); }
      });
    }

    // ---- founditem.html (report a FOUND item + match to lost owners) -----
    if (page.endsWith("founditem.html")) {
      const form = document.querySelector("form");
      form.addEventListener("submit", async function (e) {
        e.preventDefault();
        const fd = new FormData();
        fd.append("foundItemName", document.getElementById("foundItemName").value);
        fd.append("foundItemColor", document.getElementById("foundItemColor").value);
        fd.append("foundItemLocation", document.getElementById("foundItemLocation").value);
        fd.append("foundItemDate", document.getElementById("foundItemDate").value);
        fd.append("foundItemDetail", document.getElementById("foundItemDetail").value);
        fd.append("foundItemContact", document.getElementById("foundItemContact").value);
        const files = document.getElementById("foundItemImages").files;
        for (let i = 0; i < files.length; i++) fd.append("images", files[i]);

        flash(form, "กำลังค้นหาเจ้าของที่ตามหาของชิ้นนี้...", "info");
        try {
          const data = await api("/api/found-items", { method: "POST", body: fd, auth: true, form: true });
          renderPersonMatches(form, data.found_report_id, data.matches, "รายงานของหาย");
        } catch (err) { flash(form, err.message, "danger"); }
      });
    }

    // ---- data.html (public gallery of lost items + persons) --------------
    if (page.endsWith("data.html")) {
      const display = document.getElementById("data-display");
      const bar = document.getElementById("search-bar");
      const btn = document.getElementById("search-btn");
      let all = [];
      const paint = function (q) {
        const term = (q || "").trim().toLowerCase();
        const rows = !term ? all : all.filter(function (r) {
          return ((r.name || "") + " " + (r.detail || "") + " " + (r.location || "")).toLowerCase().indexOf(term) >= 0;
        });
        renderFeed(display, rows);
      };
      api("/api/feed").then(function (data) { all = data; paint(""); })
        .catch(function (e) { display.innerHTML = '<p class="text-danger">โหลดข้อมูลไม่ได้: ' + e.message + "</p>"; });
      if (btn) btn.addEventListener("click", function (e) { e.preventDefault(); paint(bar.value); });
      if (bar) bar.addEventListener("input", function () { paint(bar.value); });
    }
  });

  // render the public feed (lost items + persons) into the gallery grid
  function renderFeed(display, rows) {
    if (!rows || !rows.length) {
      display.innerHTML = '<p class="text-muted">ยังไม่มีรายการ</p>';
      return;
    }
    display.innerHTML = rows.map(function (r) {
      const done = (r.status === "success" || r.status === "matched" || r.status === "closed");
      const media = r.image_url
        ? '<img src="' + r.image_url + '" class="report-card-img" alt="">'
        : '<div class="report-media-placeholder"></div>';
      const badges = { "person-found": "พบบุคคล", "person": "บุคคลหาย", "item-found": "พบสิ่งของ", "item": "ของหาย" };
      const badge = badges[r.kind] || "สิ่งของ";
      return '<div class="col-12 col-sm-6 col-lg-4"><div class="report-card' +
        (done ? " report-card-success" : "") + '">' +
        '<div class="report-card-media">' + media + "</div>" +
        '<div class="report-card-body text-start">' +
        '<p class="report-card-line"><span class="report-card-label">' + badge + " :</span> " + escapeHtml(r.name || "-") + "</p>" +
        '<p class="report-card-line"><span class="report-card-label">รายละเอียด :</span> ' + escapeHtml(r.detail || "-") + "</p>" +
        (r.location ? '<p class="report-card-line"><span class="report-card-label">สถานที่ :</span> ' + escapeHtml(r.location) + "</p>" : "") +
        '<div class="report-status ' + (done ? "status-success" : "status-progress") + '"><span>' +
        (done ? "success" : "in progress") + "</span></div>" +
        "</div></div></div>";
    }).join("");
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // when logged in, turn the navbar "Login" button into "<name> · ออกจากระบบ"
  function updateNavbar() {
    if (!getToken()) return;
    let loginLink = null;
    document.querySelectorAll(".navbar a").forEach(function (a) {
      if ((a.textContent || "").trim().toLowerCase() === "login") loginLink = a;
    });
    if (!loginLink) return;
    api("/api/me", { auth: true }).then(function (u) {
      loginLink.textContent = (u.name || u.email || "บัญชี") + " · ออกจากระบบ";
      loginLink.href = "#";
      loginLink.addEventListener("click", function (e) { e.preventDefault(); window.upfoundLogout(); });
    }).catch(function () { clearToken(); });
  }

  // render matched person reports (lost↔found) as cards with the photo
  function renderPersonMatches(formEl, reportId, matches, labelKind) {
    const box = formEl.querySelector(".uf-flash") || (function () {
      const b = document.createElement("div"); b.className = "uf-flash mt-3"; formEl.appendChild(b); return b;
    })();
    if (!matches || !matches.length) {
      box.innerHTML = '<div class="alert alert-warning mb-0">บันทึกสำเร็จ (รายงาน #' + reportId +
        ') — ยังไม่พบ' + labelKind + 'ที่ตรงกัน (ต้องมีรูปทั้งสองฝั่งถึงจับคู่ได้)</div>';
      return;
    }
    let html = '<div class="alert alert-success">บันทึกสำเร็จ (รายงาน #' + reportId +
      ') — พบ ' + matches.length + ' ' + labelKind + 'ที่อาจตรงกัน:</div><div class="row g-3">';
    matches.forEach(function (m) {
      const pct = Math.round((m.score || 0) * 100);
      html += '<div class="col-6 col-md-4"><div class="card h-100">' +
        (m.image_url ? '<img src="' + m.image_url + '" class="card-img-top" style="object-fit:cover;height:140px">' : "") +
        '<div class="card-body p-2">' +
        '<div class="fw-500">' + escapeHtml(m.name || "ไม่ทราบชื่อ") + "</div>" +
        (m.location ? '<div class="small text-muted">' + escapeHtml(m.location) + "</div>" : "") +
        '<div class="small">ความคล้าย ' + pct + "%</div>" +
        (m.contact ? '<div class="small">ติดต่อ: ' + escapeHtml(m.contact) + "</div>" : "") +
        "</div></div></div>";
    });
    html += "</div>";
    box.innerHTML = html;
  }

  // render matched detected-items as cards with the crop image
  function renderMatches(formEl, data) {
    const box = formEl.querySelector(".uf-flash") || (function () {
      const b = document.createElement("div"); b.className = "uf-flash mt-3"; formEl.appendChild(b); return b;
    })();
    if (!data.matches || !data.matches.length) {
      box.innerHTML = '<div class="alert alert-warning mb-0">แจ้งของหายสำเร็จ (รายงาน #' + data.report_id +
        ') แต่ยังไม่พบสิ่งของที่ตรงกันในระบบ</div>';
      return;
    }
    let html = '<div class="alert alert-success">แจ้งของหายสำเร็จ (รายงาน #' + data.report_id +
      ') — พบ ' + data.matches.length + ' รายการที่อาจตรงกัน:</div><div class="row g-3">';
    data.matches.forEach(function (m) {
      const pct = Math.round(m.score * 100);
      html += '<div class="col-6 col-md-4"><div class="card h-100">' +
        (m.crop_url ? '<img src="' + m.crop_url + '" class="card-img-top" style="object-fit:cover;height:140px">' : "") +
        '<div class="card-body p-2"><div class="fw-500">' + (m.object_class || "?") + "</div>" +
        '<div class="small text-muted">' + (m.zone || "") + "</div>" +
        '<div class="small">ความคล้าย ' + pct + "%</div>" +
        '<div class="small text-muted">' + (m.capture_ts || "").slice(0, 19) + "</div>" +
        "</div></div></div>";
    });
    html += "</div>";
    box.innerHTML = html;
  }
})();
