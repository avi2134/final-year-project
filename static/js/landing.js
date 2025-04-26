document.addEventListener("DOMContentLoaded", function () {
  fetch("/api/stats/")
    .then(res => res.json())
    .then(data => {
      animateCounter("counter-users", data.users);
      animateCounter("counter-xp", data.total_xp);
    });

  function animateCounter(id, endValue) {
    const element = document.getElementById(id);
    let start = 0;
    const duration = 100;
    const stepTime = Math.max(Math.floor(duration / endValue), 20);

    const interval = setInterval(() => {
      start += 1;
      element.textContent = start.toLocaleString();
      if (start >= endValue) clearInterval(interval);
    }, stepTime);
  }
});