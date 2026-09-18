(() => {
  const dialog = document.getElementById("cv-dialog");
  if (!dialog) return;

  document.querySelectorAll("[data-cv-open]").forEach((button) => {
    button.addEventListener("click", () => {
      dialog.showModal();
      document.body.classList.add("cv-modal-open");
    });
  });

  dialog.querySelector("[data-cv-close]").addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => document.body.classList.remove("cv-modal-open"));
  dialog.addEventListener("click", (event) => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right ||
        event.clientY < bounds.top || event.clientY > bounds.bottom) {
      dialog.close();
    }
  });
})();
