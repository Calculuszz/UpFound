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
          location.href = "formitem.html";
        } catch (err) { flash(form, err.message, "danger"); }
      });
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
          location.href = "formitem.html";
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
  });

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
