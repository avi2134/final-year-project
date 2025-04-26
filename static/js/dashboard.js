let marketChart = null;
let addSymbol = null;

document.addEventListener("DOMContentLoaded", function () {
    fetchNews();             // General news
    fetchRelatedNews();      // Stock-specific carousel news
    fetchMarketSummary();    // Indices like S&P, Nasdaq, Dow
    fetchGainersAndLosers(); // Dynamic gainers/losers
    setupMarketClickForSparkline();
});

// Fetch stock-specific news
function fetchRelatedNews() {
    const symbol = document.querySelector("h3")?.textContent?.split(" ")[0] || "AAPL";

    fetch(`/fetch-news/?query=${symbol}`)
        .then(res => res.json())
        .then(data => {
            const carouselInner = document.getElementById("related-news-carousel");
            carouselInner.innerHTML = "";

            if (!data.articles.length) {
                carouselInner.innerHTML = `
                    <div class="carousel-item active">
                      <div class="d-flex align-items-center justify-content-center h-100">
                        <p class="text-muted">No related news found.</p>
                      </div>
                    </div>`;
                return;
            }

            data.articles.forEach((article, index) => {
                const item = document.createElement("div");
                item.className = `carousel-item${index === 0 ? " active" : ""}`;
                item.innerHTML = `
                  <div class="d-flex flex-column justify-content-center h-100">
                    <p><a href="${article.url}" target="_blank">${article.title}</a></p>
                  </div>
                `;
                carouselInner.appendChild(item);
            });
        })
        .catch(() => {
            document.getElementById("related-news-carousel").innerHTML = `
                <div class="carousel-item active">
                  <div class="d-flex align-items-center justify-content-center h-100">
                    <p class="text-muted">Error loading news.</p>
                  </div>
                </div>`;
        });
}

// General financial headlines
function fetchNews() {
    fetch("/fetch-news/")
        .then(res => res.json())
        .then(data => {
            const list = document.getElementById("news-list");
            list.innerHTML = "";

            if (!data.articles.length) {
                list.innerHTML = "<li class='list-group-item text-warning'>No news found.</li>";
                return;
            }

            data.articles.forEach(article => {
                const li = document.createElement("li");
                li.classList.add("list-group-item");
                li.innerHTML = `<a href="${article.url}" target="_blank">${article.title}</a>`;
                list.appendChild(li);
            });
        })
        .catch(() => {
            document.getElementById("news-list").innerHTML = "<li class='list-group-item'>Error loading news.</li>";
        });
}

// Major market index data
function fetchMarketSummary() {
    fetch("/fetch-market-summary/")
        .then(res => res.json())
        .then(data => {
            const marketList = document.getElementById("market-summary");
            marketList.innerHTML = "";

            if (data.summary && data.summary.length) {
                data.summary.forEach(item => {
                    const li = document.createElement("li");
                    li.classList.add("list-group-item", "d-flex", "justify-content-between", "align-items-center");

                    const changeColor = item.is_up ? "text-success" : "text-danger";
                    const arrow = item.is_up ? "▲" : "▼";
                    const changeText = `${arrow} ${item.change > 0 ? "+" : ""}${item.change} (${item.percent_change}%)`;

                    li.innerHTML = `
                        <div><strong>${item.name}</strong>: $${item.price}</div>
                        <div class="${changeColor}">${changeText}</div>
                    `;

                    marketList.appendChild(li);
                });
            } else {
                marketList.innerHTML = "<li class='list-group-item text-warning'>No index data available.</li>";
            }
        })
        .catch(() => {
            document.getElementById("market-summary").innerHTML =
                "<li class='list-group-item text-danger'>Error loading market summary</li>";
        });
}

function setupMarketClickForSparkline() {
    const observer = new MutationObserver(() => {
        const items = document.querySelectorAll("#market-summary .list-group-item");
        if (items.length > 0) {
            observer.disconnect();

            items.forEach((item, idx) => {
                const name = item.querySelector("strong")?.textContent?.trim();
                const symbol = getMarketSymbolByName(name);

                if (!symbol) return;

                item.dataset.symbol = symbol;
                item.style.cursor = "pointer";

                if (idx === 0) {
                    item.classList.add("active");
                    fetchSparklineChart(symbol);
                }

                item.addEventListener("click", function () {
                    document.querySelectorAll("#market-summary .list-group-item").forEach(el => el.classList.remove("active"));
                    this.classList.add("active");
                    fetchSparklineChart(this.dataset.symbol);
                });
            });
        }
    });

    observer.observe(document.getElementById("market-summary"), { childList: true });
}

function getMarketSymbolByName(name) {
    const mapping = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",
        "VIX": "^VIX",
        "Nikkei 225": "^N225"
    };
    return mapping[name] || null;
}

function fetchSparklineChart(symbol) {
    fetch(`/fetch-market-chart/?symbol=${encodeURIComponent(symbol)}`)
        .then(res => res.json())
        .then(data => {
            const activeItem = document.querySelector("#market-summary .list-group-item.active");
            const name = activeItem.querySelector("strong")?.textContent?.trim();
            document.getElementById("sparkline-title").textContent = `5-Day Price Trend: ${name}`;
            const prices = data.prices;
            if (!prices || prices.length === 0) return;

            const ctx = document.getElementById("market-sparkline").getContext("2d");
            if (marketChart) marketChart.destroy();

            marketChart = new Chart(ctx, {
                type: "line",
                data: {
                    labels: prices.map((_, i) => i + 1),
                    datasets: [{
                        data: prices,
                        borderColor: "#0d6efd",
                        backgroundColor: "rgba(13, 110, 253, 0.1)",
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                               tooltips: { display: true },
                    scales: {
                        x: { display: true },
                        y: { display: true }
                    }
                }
            });
        });
}

// Fetch top gainers and losers
function fetchGainersAndLosers() {
    fetch("/fetch-gainers-losers/")
        .then(res => res.json())
        .then(data => {
            const glList = document.getElementById("gainers-losers");
            glList.innerHTML = "";

            const gainers = data.gainers || [];
            const losers = data.losers || [];

            glList.innerHTML += "<li class='list-group-item fw-bold text-success'>Top Gainers</li>";
            gainers.forEach(stock => {
                const li = document.createElement("li");
                li.classList.add("list-group-item");
                li.innerHTML = `${stock.shortName || stock.symbol}: <span class='text-success'>+${stock.regularMarketChangePercent?.toFixed(2)}%</span>`;
                glList.appendChild(li);
            });

            glList.innerHTML += "<li class='list-group-item fw-bold text-danger'>Top Losers</li>";
            losers.forEach(stock => {
                const li = document.createElement("li");
                li.classList.add("list-group-item");
                li.innerHTML = `${stock.shortName || stock.symbol}: <span class='text-danger'>${stock.regularMarketChangePercent?.toFixed(2)}%</span>`;
                glList.appendChild(li);
            });
        })
        .catch(() => {
            document.getElementById("gainers-losers").innerHTML =
                "<li class='list-group-item text-danger'>Error fetching gainers/losers</li>";
        });
}

document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("watchlist-form").addEventListener("submit", function (e) {
    e.preventDefault();
    const input = document.getElementById("stock-search-modal");
    const symbol = input.value.trim().toUpperCase();
    const messageBox = document.getElementById("watchlist-message");

    fetch(`/fetch-stock-search/?query=${encodeURIComponent(symbol)}`)
        .then(res => res.json())
        .then(data => {
            const valid = data.bestMatches?.some(stock => stock.symbol === symbol);
            if (!valid) {
                messageBox.classList.remove("text-success");
                messageBox.classList.add("text-danger");
                messageBox.textContent = "Invalid stock symbol. Please select from the list.";
                return;
            }

            fetch("/add-to-watchlist/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-CSRFToken": document.querySelector("[name=csrfmiddlewaretoken]").value
                },
                body: JSON.stringify({ symbol: symbol })
            })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        messageBox.classList.remove("text-danger");
                        messageBox.classList.add("text-success");
                        messageBox.textContent = "Stock added successfully!";
                        setTimeout(() => location.reload(), 1000);
                    } else {
                        messageBox.classList.remove("text-success");
                        messageBox.classList.add("text-danger");
                        messageBox.textContent = data.error || "Failed to add stock.";
                    }
                })
                .catch(() => {
                    messageBox.classList.remove("text-success");
                    messageBox.classList.add("text-danger");
                    messageBox.textContent = "Something went wrong.";
                });
        });
    });
});

let deleteSymbol = null;

function deleteWatchedStock(symbol) {
  deleteSymbol = symbol;
  document.getElementById("stockToDeleteName").textContent = symbol;
  const modal = new bootstrap.Modal(document.getElementById("deleteConfirmModal"));
  modal.show();
}

document.getElementById("confirmDeleteBtn").addEventListener("click", function () {
  if (!deleteSymbol) return;

  fetch("/delete-stock/", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      "X-CSRFToken": getCookie("csrftoken"),
    },
    body: new URLSearchParams({ symbol: deleteSymbol })
  })
    .then(res => res.json())
    .then(data => {
      if (data.success) {
        const row = document.querySelector(`[data-symbol="${deleteSymbol}"]`);
        if (row) row.remove();
      } else {
        alert("Failed to remove stock.");
      }

      // Close modal
      const modalElement = document.getElementById("deleteConfirmModal");
      const modalInstance = bootstrap.Modal.getInstance(modalElement);
      modalInstance.hide();
    })
    .catch(() => alert("Something went wrong."));
});

// Get CSRF token
function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== "") {
    const cookies = document.cookie.split(";");
    for (let i = 0; i < cookies.length; i++) {
      const cookie = cookies[i].trim();
      if (cookie.startsWith(name + "=")) {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}

document.addEventListener("DOMContentLoaded", function () {
    const toggleBtn = document.getElementById("toggle-watchlist-btn");
    const hiddenRows = document.querySelectorAll(".watch-row.extra-row");

    if (toggleBtn && hiddenRows.length > 0) {
        toggleBtn.addEventListener("click", function () {
            hiddenRows.forEach(row => row.classList.toggle("d-none"));

            if (hiddenRows[0].classList.contains("d-none")) {
                toggleBtn.innerHTML = "<i class=\"bi bi-chevron-double-down\"></i> Show More";
            } else {
                toggleBtn.innerHTML = "<i class=\"bi bi-chevron-double-up\"></i> Show Less";
            }
        });
    } else if (toggleBtn) {
        toggleBtn.remove(); // No need for button if <=5 items
    }
});

function confirmAddSuggestedStock(symbol) {
  addSymbol = symbol;  // Save symbol globally
  document.getElementById("stockToAddName").textContent = symbol; // Set the modal text

  const modal = new bootstrap.Modal(document.getElementById("addConfirmModal"));
  modal.show();  // Open the modal
}

document.getElementById("confirmAddBtn").addEventListener("click", function () {
  if (!addSymbol) return;

  fetch("/add-to-watchlist/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRFToken": getCookie('csrftoken')
    },
    body: JSON.stringify({ symbol: addSymbol })
  })
  .then(res => res.json())
  .then(data => {
    if (data.success) {
      showSuccessToast(`${addSymbol} added to your watchlist!`);

      const suggestedRow = document.getElementById(`suggested-${addSymbol}`);
      if (suggestedRow) {
        // 🆕 Clone the suggested row
        const newRow = suggestedRow.cloneNode(true);

        // 🆕 Update the cloned row for Watchlist
        newRow.querySelector("td:last-child").innerHTML = `
          <button onclick="deleteWatchedStock('${addSymbol}')" class="btn btn-sm btn-outline-danger">
            Remove
          </button>
        `;
        newRow.id = `stock-${addSymbol}`;
        newRow.dataset.symbol = addSymbol;
        newRow.classList.add("watch-row");

        // 🆕 Append the cloned row to Watchlist table
        const watchlistTable = document.querySelector("table tbody");
        watchlistTable.appendChild(newRow);

        // 🆕 Remove the original suggested stock row
        suggestedRow.remove();

        // 🆕 Check if no more suggestions left
        const suggestedBody = document.getElementById("suggested-stocks-body");
        if (suggestedBody && suggestedBody.children.length === 0) {
          document.getElementById("suggested-stocks-table").classList.add("d-none");
          document.getElementById("no-suggestions-message").classList.remove("d-none");
        }
      }
    } else {
      alert(data.error || "Failed to add stock.");
    }

    // 🆕 Close the Add Confirmation Modal
    const modalElement = document.getElementById("addConfirmModal");
    const modalInstance = bootstrap.Modal.getInstance(modalElement);
    modalInstance.hide();
    addSymbol = null; // Reset after adding
  })
  .catch(() => alert("Something went wrong."));
});

function showSuccessToast(message) {
  const toastEl = document.getElementById('successToast');
  const toastBody = toastEl.querySelector('.toast-body');
  toastBody.textContent = message;

  const toast = new bootstrap.Toast(toastEl);
  toast.show();
}