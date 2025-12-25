
/* ---------- USER DROPDOWN ---------- */
function toggleUserDropdown() {
    const menu = document.getElementById("userDropdown");
    if (!menu) return;

    menu.classList.toggle("hidden");
}

/* ---------- LOGIN MODAL ---------- */
function openLoginModal() {
    const modal = document.getElementById("loginModal");
    modal?.classList.remove("hidden");
    modal?.classList.add("flex");
}

function closeLoginModal() {
    const modal = document.getElementById("loginModal");
    modal?.classList.add("hidden");
    modal?.classList.remove("flex");
}

/* ---------- REGISTER MODAL ---------- */
function openRegisterModal() {
    closeLoginModal();
    const modal = document.getElementById("registerModal");
    modal?.classList.remove("hidden");
    modal?.classList.add("flex");
}

function closeRegisterModal() {
    const modal = document.getElementById("registerModal");
    modal?.classList.add("hidden");
    modal?.classList.remove("flex");
}

/* ---------- LOGIN ---------- */
async function loginUser(e) {
    e.preventDefault();

    const username = document.getElementById("login_username").value;
    const password = document.getElementById("login_password").value;

    const res = await fetch("/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password })
    });

    const data = await res.json();

    if (data.success) {
        location.reload();
    } else {
        alert(data.msg || "Login failed");
    }
}

function logoutUser() {
    window.location.href = "/logout";
}

document.getElementById("registerForm")?.addEventListener("submit", async function (e) {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);

    const res = await fetch("/register", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    if (data.success) {
        alert("Register success! Please login.");
        closeRegisterModal();
        openLoginModal();
        form.reset();
    } else {
        alert(data.msg || "Register failed");
    }
});

function openAccountPopup(type) {
    if (type !== "profile") return;

    const popup = document.getElementById("accountPopup");
    const form = document.getElementById("profileForm");

    popup.classList.remove("hidden");

    fetch("/api/profile")
        .then(res => res.json())
        .then(data => {
            for (const key in data) {
                const field = form.querySelector(`[name="${key}"]`);
                if (field) field.value = data[key];
            }
        });
}

document.getElementById("profileForm").addEventListener("submit", function (e) {
    e.preventDefault();

    const formData = new FormData(this);
    const data = {};

    formData.forEach((value, key) => {
        if (value !== "") data[key] = value;
    });

    fetch("/api/profile/update", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
        .then(res => res.json())
        .then(res => {
            if (res.success) {
                alert("✅ Profile saved");
            } else {
                alert("❌ " + res.msg);
            }
        });
});

document.addEventListener("mousedown", function (e) {
    const userMenu = document.getElementById("userMenu");
    const dropdown = document.getElementById("userDropdown");
    const popup = document.getElementById("accountPopup");

    if (!userMenu) return;

    // ถ้าคลิกอยู่นอก userMenu ทั้งหมด
    if (!userMenu.contains(e.target)) {
        dropdown?.classList.add("hidden");
        popup?.classList.add("hidden");
    }
});
