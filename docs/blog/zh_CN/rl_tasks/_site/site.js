(function () {
  const sidebar = document.querySelector(".sidebar");
  const key = "unilab-rl-docs-sidebar-scroll";
  if (!sidebar || !window.sessionStorage) return;

  const saved = Number(window.sessionStorage.getItem(key));
  if (Number.isFinite(saved)) {
    requestAnimationFrame(() => {
      sidebar.scrollTop = saved;
    });
  }

  const saveScroll = () => {
    window.sessionStorage.setItem(key, String(sidebar.scrollTop));
  };

  sidebar.addEventListener("scroll", saveScroll, { passive: true });
  sidebar.addEventListener(
    "click",
    (event) => {
      const target = event.target instanceof Element ? event.target : null;
      if (target && target.closest("a[href]")) saveScroll();
    },
    true
  );
  window.addEventListener("pagehide", saveScroll);
})();

(async function () {
  const input = document.getElementById("search");
  const box = document.getElementById("search-results");
  if (!input || !box || !window.SEARCH_INDEX_URL) return;
  const response = await fetch(window.SEARCH_INDEX_URL);
  const pages = await response.json();
  const prefix = window.SITE_ROOT_PREFIX || "";
  input.addEventListener("input", () => {
    const q = input.value.trim().toLowerCase();
    box.innerHTML = "";
    if (!q) return;
    const hits = pages.filter((p) => (p.title + " " + p.category + " " + p.text).toLowerCase().includes(q)).slice(0, 8);
    if (!hits.length) {
      box.textContent = "没有匹配结果";
      return;
    }
    const list = document.createElement("ul");
    for (const hit of hits) {
      const item = document.createElement("li");
      const link = document.createElement("a");
      const title = document.createElement("strong");
      const meta = document.createElement("small");
      link.href = prefix + hit.href;
      title.textContent = hit.title;
      meta.textContent = hit.category;
      link.appendChild(title);
      link.appendChild(meta);
      item.appendChild(link);
      list.appendChild(item);
    }
    box.appendChild(list);
  });
})();

(function () {
  const toc = document.querySelector(".page-toc");
  if (!toc) return;
  const links = Array.from(toc.querySelectorAll("a[href^='#']"));
  const headings = links
    .map((link) => document.getElementById(decodeURIComponent(link.getAttribute("href").slice(1))))
    .filter(Boolean);
  if (!links.length || !headings.length) return;

  const setActive = () => {
    let active = headings[0];
    for (const heading of headings) {
      if (heading.getBoundingClientRect().top <= 140) {
        active = heading;
      } else {
        break;
      }
    }
    const id = active.id;
    for (const link of links) {
      const isActive = link.getAttribute("href") === "#" + id;
      link.classList.toggle("active", isActive);
      if (isActive) link.scrollIntoView({ block: "nearest" });
    }
  };

  setActive();
  window.addEventListener("scroll", setActive, { passive: true });
})();
