document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".stock-search-wrapper").forEach((wrapper) => {
    const context = wrapper.dataset.context || "main";
    const input = wrapper.querySelector(`#stock-search-${context}`);
    const dropdown = wrapper.querySelector(`#search-results-${context}`);

    if (!input || !dropdown) return;

    input.addEventListener("input", function () {
      const query = this.value.trim();

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

              item.addEventListener("mouseover", () => item.classList.add("active"));
              item.addEventListener("mouseout", () => item.classList.remove("active"));
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

    document.addEventListener("click", function (e) {
      if (!wrapper.contains(e.target)) {
        dropdown.innerHTML = "";
        dropdown.classList.add("d-none");
      }
    });
  });
});
