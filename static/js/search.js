document.getElementById("stock-search").addEventListener("input", function() {
    let query = this.value.trim();
    if (query.length < 2) return;

    fetch(`/fetch-stock-search/?query=${query}`)
        .then(response => response.json())
        .then(data => {
            let dropdown = document.getElementById("search-results");
            dropdown.innerHTML = ""; // Clear previous results

            if (data.bestMatches && data.bestMatches.length > 0) {
                data.bestMatches.forEach(stock => {
                    let item = document.createElement("li");
                    item.classList.add("list-group-item");
                    item.innerHTML = `
                        <strong>${stock.symbol}</strong> - ${stock.name} 
                        <span class="badge bg-secondary">${stock.exchange}</span>
                    `;
                    item.onclick = () => {
                        document.getElementById("stock-search").value = stock.symbol; // Autofill input with selected stock symbol
                        dropdown.innerHTML = ""; // Clear dropdown after selection
                    };
                    dropdown.appendChild(item);
                });
            } else {
                dropdown.innerHTML = `<li class="list-group-item text-muted">No results found</li>`;
            }
        })
        .catch(error => console.error("Error fetching stock search:", error));
});