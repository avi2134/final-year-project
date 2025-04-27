document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".stock-search-wrapper").forEach((wrapper) => {
    const context = wrapper.dataset.context || "main";
    const input = wrapper.querySelector(`#stock-search-${context}`);
    const dropdown = wrapper.querySelector(`#search-results-${context}`);
    let currentIndex = -1;

    if (!input || !dropdown) return;

    input.addEventListener("input", function () {
      const query = this.value.trim();
      currentIndex = -1;

      if (query.length < 2) {
        dropdown.innerHTML = "";
        dropdown.classList.add("d-none");
        return;
      }

      fetch(`/fetch-stock-search/?query=${encodeURIComponent(query)}`)
        .then((res) => res.json())
        .then((data) => {
          dropdown.innerHTML = "";

          if (data.bestMatches?.length) {
            data.bestMatches.forEach((stock) => {
              const item = document.createElement("li");
              item.classList.add("list-group-item", "result-item");

              item.innerHTML = `
                <strong>${stock.symbol}</strong> - ${stock.name}
                <span class="badge bg-secondary">${stock.exchange}</span>
              `;

              item.addEventListener("click", () => {
                input.value = stock.symbol;
                dropdown.innerHTML = "";
                dropdown.classList.add("d-none");
              });

              dropdown.appendChild(item);
            });

            dropdown.classList.remove("d-none");
          } else {
            dropdown.innerHTML = `<li class="list-group-item text-muted">No results found</li>`;
            dropdown.classList.remove("d-none");
          }
        })
        .catch((err) => {
          console.error("Search error:", err);
          dropdown.innerHTML = `<li class="list-group-item text-danger">Error loading results</li>`;
          dropdown.classList.remove("d-none");
        });
    });

    // 🧠 Keyboard navigation
    input.addEventListener("keydown", function (e) {
      const items = dropdown.querySelectorAll(".result-item");
      if (items.length === 0 || dropdown.classList.contains("d-none")) return;

      if (e.key === "ArrowDown") {
        e.preventDefault();
        currentIndex = (currentIndex + 1) % items.length;
        updateActive(items);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        currentIndex = (currentIndex - 1 + items.length) % items.length;
        updateActive(items);
      } else if (e.key === "Enter") {
        if (currentIndex >= 0 && items[currentIndex]) {
          e.preventDefault(); // important!
          items[currentIndex].click();
        }
      }
    });

    function updateActive(items) {
      items.forEach((item) => item.classList.remove("active"));
      if (currentIndex >= 0 && items[currentIndex]) {
        items[currentIndex].classList.add("active");
        items[currentIndex].scrollIntoView({ block: "nearest" });
      }
    }

    // Hide dropdown on click outside
    document.addEventListener("click", function (e) {
      if (!wrapper.contains(e.target)) {
        dropdown.innerHTML = "";
        dropdown.classList.add("d-none");
      }
    });
  });
});