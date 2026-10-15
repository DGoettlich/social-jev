// The website owns the manifest and decides which release is latest stable.
document.addEventListener("DOMContentLoaded", async () => {
  const select = document.querySelector("[data-docs-version]");
  const status = document.getElementById("version-status");
  const match = location.pathname.match(/^\/docs\/(v[^/]+)(?:\/|$)/);
  const current = match ? match[1] : select.dataset.docsVersion;
  try {
    const response = await fetch("/docs/versions.json");
    if (!response.ok) throw new Error("Version manifest unavailable");
    const manifest = await response.json();
    const versions = manifest.versions.includes(current)
      ? manifest.versions : [current, ...manifest.versions];
    select.replaceChildren(...versions.map(version => {
      const label = version === manifest.latest ? `${version} (latest)` : version;
      return new Option(label, version, false, version === current);
    }));
  } catch {
    status.textContent = "Published version list unavailable.";
  }
  select.addEventListener("change", async () => {
    const root = `/docs/${encodeURIComponent(select.value)}/`;
    const currentRoot = `/docs/${current}/`;
    const page = location.pathname.startsWith(currentRoot)
      ? location.pathname.slice(currentRoot.length) : "";
    select.disabled = true;
    try {
      const target = root + page;
      const response = await fetch(target, {method: "HEAD"});
      location.assign(response.ok ? target + location.hash : root);
    } catch {
      select.value = current;
      select.disabled = false;
      status.textContent = "Could not switch documentation versions. Please try again.";
    }
  });
});
