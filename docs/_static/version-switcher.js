/**
 * Version switcher for Furo theme.
 *
 * Detects the site base path from the current URL (handles GitLab Pages
 * subpath deployments like /project-name/) and fetches versions.json
 * relative to it.
 */
document.addEventListener("DOMContentLoaded", function () {
  var path = window.location.pathname;

  // Detect base path: everything before /latest/ or /v0.7.6/ etc.
  var match = path.match(/^(.*?)\/(latest|v[0-9.]+)\//);
  if (!match) return;
  var basePath = match[1];

  fetch(basePath + "/versions.json")
    .then(function (r) { return r.json(); })
    .then(function (versions) {
      if (!versions || versions.length < 2) return;

      var select = document.createElement("select");
      select.setAttribute("aria-label", "Version selector");
      select.style.cssText =
        "margin:0.5rem 0;padding:4px 8px;width:100%;border:1px solid var(--color-sidebar-link-text);border-radius:4px;background:var(--color-sidebar-background);color:var(--color-sidebar-link-text);font-size:0.85rem;cursor:pointer;";

      versions.forEach(function (v) {
        var opt = document.createElement("option");
        opt.value = basePath + v.url;
        opt.textContent = v.name;
        if (path.indexOf(basePath + v.url) === 0) {
          opt.selected = true;
        }
        select.appendChild(opt);
      });

      select.addEventListener("change", function () {
        window.location.href = select.value;
      });

      var container = document.querySelector(".sidebar-brand");
      if (container) {
        container.appendChild(select);
      }
    })
    .catch(function () {});
});
