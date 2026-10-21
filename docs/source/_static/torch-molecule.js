document.addEventListener("DOMContentLoaded", () => {
  const menu = document.querySelector(".tm-menu-toggle");
  const navigation = document.querySelector(".tm-navigation");
  const search = document.querySelector(".tm-search input");
  const searchForm = document.querySelector(".tm-search");
  const searchLink = document.createElement("a");
  searchLink.href = searchForm.action;
  searchLink.textContent = "Search documentation";
  searchLink.className = "tm-mobile-search-link";
  searchLink.style.display = "none";
  navigation.append(searchLink);

  const setMenu = (open) => {
    menu.setAttribute("aria-expanded", String(open));
    navigation.dataset.open = String(open);
    searchLink.style.display = open ? "block" : "none";
  };
  menu.addEventListener("click", () => setMenu(menu.getAttribute("aria-expanded") !== "true"));
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      if (search.offsetParent) search.focus();
      else window.location.href = searchForm.action;
    }
    if (event.key === "Escape" && menu.getAttribute("aria-expanded") === "true") {
      setMenu(false);
      menu.focus();
    }
  });
  if (!/Mac|iPhone|iPad/.test(navigator.platform)) document.querySelector(".tm-search kbd").textContent = "Ctrl K";

  const tabs = [...document.querySelectorAll('.tm-code-tabs [role="tab"]')];
  const activateTab = (selected) => {
    tabs.forEach((tab) => {
      const active = tab === selected;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      document.getElementById(tab.getAttribute("aria-controls")).hidden = !active;
    });
  };
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activateTab(tab));
    tab.addEventListener("keydown", (event) => {
      const direction = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      let target;
      if (direction) target = tabs[(index + direction + tabs.length) % tabs.length];
      else if (event.key === "Home") target = tabs[0];
      else if (event.key === "End") target = tabs[tabs.length - 1];
      if (target) {
        event.preventDefault();
        activateTab(target);
        target.focus();
      }
    });
  });
  if (tabs.length) activateTab(tabs[0]);

  const status = document.getElementById("tm-copy-status") || document.body.appendChild(Object.assign(document.createElement("div"), { className: "tm-sr-only" }));
  status.setAttribute("role", "status");
  const copy = async (button, text) => {
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "Copied";
      status.textContent = "Copied to clipboard.";
    } catch {
      button.textContent = "Select & copy";
      status.textContent = "Clipboard is unavailable. Select the code to copy it.";
    }
    window.setTimeout(() => { button.textContent = "Copy"; }, 2000);
  };
  document.querySelectorAll("[data-copy-text]").forEach((button) => {
    button.addEventListener("click", () => copy(button, button.dataset.copyText));
  });
  document.querySelector(".tm-code-copy")?.addEventListener("click", (event) => {
    copy(event.currentTarget, document.querySelector('.tm-code-window [role="tabpanel"]:not([hidden]) code').textContent);
  });
  document.querySelectorAll("article .highlight pre").forEach((pre) => {
    const button = document.createElement("button");
    button.className = "tm-doc-copy";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy code block");
    button.addEventListener("click", () => copy(button, pre.textContent));
    pre.parentElement.append(button);
  });
  document.querySelectorAll("article pre, article table.docutils").forEach((element) => { element.tabIndex = 0; });
});
