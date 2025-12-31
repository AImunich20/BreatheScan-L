
(() => {
    const btn = document.getElementById("mobile-menu-button");
    const menu = document.getElementById("mobile-menu");

    if (!btn || !menu) {
        console.warn("❌ mobile menu elements not found");
        return;
    }

    btn.addEventListener("click", () => {
        console.log("✅ mobile menu clicked");
        menu.classList.toggle("hidden");
    });
})();

