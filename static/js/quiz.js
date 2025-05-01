document.addEventListener("DOMContentLoaded", function () {
    let quizContainer = document.getElementById("quiz-container");
    let submitButton = document.getElementById("submit-quiz");
    let nextLevelButton = document.getElementById("next-level");
    let xpBar = document.getElementById("xp-bar");
    let xpPoints = document.getElementById("xp-points");
    let xpNeededLabel = document.getElementById("xp-needed-label");
    let currentQuiz = null; // Store active quiz
    let alertMessage = document.getElementById("alert");

    function getCSRFToken() {
        return document.cookie.split('; ').find(row => row.startsWith('csrftoken='))?.split('=')[1];
    }

    function levelUp() {
        fetch("/api/level-up/")
        .then(() => location.reload());
    }

    function fetchQuizQuestions(quizNumber) {
        currentQuiz = quizNumber; // Store current quiz number
        fetch(`/api/get-quiz-questions/?quiz=${quizNumber}`)
            .then(response => response.json())
            .then(data => {
                if (data.message) {
                    quizContainer.innerHTML = `<p class="text-danger">${data.message}</p>`;
                    submitButton.style.display = "none";
                    return;
                }

                quizContainer.innerHTML = "";
                submitButton.style.display = "block";

                data.questions.forEach((q, index) => {
                    let questionHTML = `<div class="card p-3 mb-2">
                        <p><strong>${index + 1}. ${q.question}</strong></p>
                        <div>
                            <input type="radio" name="question${q.id}" value="A"> ${q.options.A} <br>
                            <input type="radio" name="question${q.id}" value="B"> ${q.options.B} <br>
                            <input type="radio" name="question${q.id}" value="C"> ${q.options.C} <br>
                            <input type="radio" name="question${q.id}" value="D"> ${q.options.D} <br>
                        </div>
                    </div>`;
                    quizContainer.innerHTML += questionHTML;
                });
            })
            .catch(error => {
                quizContainer.innerHTML = `<p class="text-danger">Error loading quiz. Please try again.</p>`;
                console.error("Error fetching quiz questions:", error);
            });
    }

    document.querySelectorAll('.quiz-button').forEach(button => {
        button.addEventListener('click', function () {
            let quizNumber = this.dataset.quiz;
            fetchQuizQuestions(quizNumber);
        });
    });

    submitButton.addEventListener("click", function () {
        let answers = {};
        document.querySelectorAll("input[type=radio]:checked").forEach(input => {
            answers[input.name.replace("question", "")] = input.value;
        });

        fetch(`/api/submit-quiz-answers/?quiz=${currentQuiz}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
                "X-CSRFToken": getCSRFToken()
            },
            body: new URLSearchParams(answers),
        })
        .then(response => response.json())
        .then(data => {
            // Calculate XP percentage out of 50
            let xpPercentage = (data.xp_gained / 50) * 100;

            // Show animated modal XP bar
            showXPProgress(xpPercentage);

            // Show modal
            let modal = new bootstrap.Modal(document.getElementById('exampleModal'));
            modal.show();

            // Update current quiz card XP
            if (currentQuiz && data.xp_gained !== undefined) {
                let quizXpElement = document.getElementById(`quiz-xp-${currentQuiz}`);
                if (quizXpElement) {
                    quizXpElement.textContent = `XP: ${data.xp_gained} / 50`;
                }
            }

            // Update XP bar and level
            fetch('/api/get-user-progress/')
                .then(response => response.json())
                .then(progress => {
                    let xp = progress.xp_points;
                    let xpNeeded = progress.xp_needed;
                    let maxXP = 800;
                    let xpBarWidth = (xp / maxXP) * 100;
                    let levelIndicator = (xpNeeded / maxXP) * 100;

                    xpBar.style.width = `${xpBarWidth}%`;
                    xpPoints.innerText = `${xp} XP / ${xpNeeded} XP to Level Up`;
                    document.getElementById("quiz-level").innerText = progress.level;

                    xpNeededLabel.innerHTML = `<div style="position:absolute; left:${levelIndicator}%; top: -10px; font-size: 14px;">⬆️ ${xpNeeded} XP</div>`;

                    if (xp >= xpNeeded) {
                        nextLevelButton.style.display = "block";
                        alertMessage.style.display = "block";
                    } else {
                        nextLevelButton.style.display = "none";
                        alertMessage.style.display = "none";
                    }

                    // Update all quiz cards to reflect latest XP
                    let xpScores = progress.quiz_xp || {};
                    for (let i = 1; i <= 4; i++) {
                        let el = document.getElementById(`quiz-xp-${i}`);
                        if (el) {
                            el.textContent = `XP: ${xpScores[i] || 0} / 50`;
                        }
                    }
                });
        })
        .catch(error => {
            console.error("Error submitting quiz answers:", error);
            alert("Something went wrong submitting your quiz. Please try again.");
        });
    });

    // The XPbar code has been modified from ProgressBar.js 1.1.1 (https://kimmobrunfeldt.github.io/progressbar.js)
    function showXPProgress(xpPercentage) {
        let container = document.getElementById("container");
        container.innerHTML = ""; // Clear previous progress bar

        let bar = new ProgressBar.Circle(container, {
        strokeWidth: 5,
        trailWidth: 2,
        easing: 'easeInOut',
        duration: 1400,
        text: {
            autoStyleContainer: false
        },
        from: { color: '#FC5B3F', width: 2 },  // Start Color (Red)
        to: { color: '#6DD47E', width: 6 },   // End Color (Green)
        svgStyle: { width: "200px", height: "200px" },
        step: function(state, circle) {
            let progressValue = Math.round(circle.value() * 100);

            // Change the text to show XP %
            circle.setText(progressValue + "% XP Earned");

            // Dynamic color transition from Red → Yellow → Green
            if (progressValue <= 20) {
                circle.path.setAttribute('stroke', '#FC5B3F'); // Red
            } else if (progressValue <= 60) {
                circle.path.setAttribute('stroke', '#FFA500'); // Orange/Yellow
            } else {
                circle.path.setAttribute('stroke', '#6DD47E'); // Green
            }
        }
    });

    bar.text.style.fontFamily = '"Raleway", Helvetica, sans-serif';
    bar.text.style.fontSize = '1.3rem';

    bar.animate(xpPercentage / 100); // Animate based on XP gained
    }

    function updateXPBar() {
        fetch('/api/get-user-progress/')
            .then(response => response.json())
            .then(data => {
                let xp = data.xp_points;
                let xpNeeded = data.xp_needed;
                let maxXP = 800;
                let xpPercentage = (xp / maxXP) * 100;
                let levelPercentage = (xpNeeded / maxXP) * 100;

                xpBar.style.width = `${xpPercentage}%`;
                xpPoints.innerText = `${xp} XP / ${xpNeeded} XP to Level Up`;
                document.getElementById("quiz-level").innerText = data.level;

                xpNeededLabel.innerHTML = `<div style="position:absolute; left:${levelPercentage}%; top: -10px; font-size: 14px;">⬆️ ${xpNeeded} XP</div>`;

                if (xp >= xpNeeded) {
                    nextLevelButton.style.display = "block";
                    alertMessage.style.display = "block";
                } else {
                    nextLevelButton.style.display = "none";
                    alertMessage.style.display = "none";
                }
            })
            .catch(error => console.error("Error fetching XP data:", error));
    }

    nextLevelButton.addEventListener("click", function () {
        fetch("/api/level-up/", {
            method: "POST",
            headers: {
                "X-CSRFToken": getCSRFToken()
            }
        })
        .then(() => location.reload());
    });

    function fetchQuizXP() {
        fetch("/api/get-user-progress/")
            .then(response => response.json())
            .then(data => {
                let xpScores = data.quiz_xp || {};
                for (let i = 1; i <= 4; i++) {
                    let quizXpElement = document.getElementById(`quiz-xp-${i}`);
                    if (quizXpElement) {
                        let xp = xpScores[i] || 0; // Default to 0 XP if no data
                        quizXpElement.textContent = `XP: ${xp} / 50`;
                    }
                }

            let levelBadge = document.getElementById("level-badge");
            let badgeURL = "";
                if (data.level === "beginner") {
                    badgeURL = "https://img.icons8.com/fluency/100/favorites-shield.png";
                } else if (data.level === "intermediate") {
                    badgeURL = "https://img.icons8.com/fluency/100/favorites-shield-2.png";
                } else if (data.level === "advanced") {
                    badgeURL = "https://img.icons8.com/fluency/100/favorites-shield-3.png";
                } else if (data.level === "expert") {
                    badgeURL = "https://img.icons8.com/fluency/100/favorites-shield-4.png";
                }

                levelBadge.src = badgeURL;
            })
            .catch(error => console.error("Error fetching quiz XP:", error));
    }

    function fetchQuizHistory() {
        fetch('/api/get-quiz-history/')
            .then(response => response.json())
            .then(data => {
                let historyContainer = document.getElementById("quizHistory");
                historyContainer.innerHTML = ""; // Clear previous entries

                if (!data.completed_levels) {
                    historyContainer.innerHTML = `<div class="alert alert-warning" role="alert"><i class="bi bi-info-circle-fill me-2"></i> No levels completed yet. Finish a level to see quiz history!</div>`;
                    return;
                }

                let levels = Object.keys(data.history);
                levels.forEach(level => {
                    if (data.history[level].length === 0) return; // Skip empty levels

                    let levelHTML = `
                    <div class="mt-4">
                        <h4 class="text-capitalize">${level}</h4>
                        <div class="accordion" id="accordion-${level}">
                    `;

                    data.history[level].forEach((quiz, index) => {
                        levelHTML += `
                        <div class="accordion-item">
                            <h2 class="accordion-header" id="heading-${level}-${index}">
                                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse-${level}-${index}" aria-expanded="false" aria-controls="collapse-${level}-${index}">
                                    Quiz ${quiz.quiz_number}
                                </button>
                            </h2>
                            <div id="collapse-${level}-${index}" class="accordion-collapse collapse" aria-labelledby="heading-${level}-${index}" data-bs-parent="#accordion-${level}">
                                <div class="accordion-body">
                                    ${quiz.questions.map(q => `
                                        <div class="card p-3 mb-2">
                                            <p><strong>Q: ${q.question}</strong></p>
                                            <p>Correct Answer: <strong>${q.correct_answer}</strong></p>
                                            <p><em>Explanation: ${q.explanation}</em></p>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        </div>`;
                    });

                    levelHTML += `</div></div>`;
                    historyContainer.innerHTML += levelHTML;
                });
            })
            .catch(error => console.error("Error fetching quiz history:", error));
    }

    fetchQuizHistory();
    fetchQuizXP();
    updateXPBar();
});

// progressbar.js@1.0.0 version is used
// Docs: http://progressbarjs.readthedocs.org/en/1.0.0/

var ProgressBar = window.ProgressBar;
var bar = new ProgressBar.Circle(container, {
  color: '#aaa',
  // This has to be the same size as the maximum width to
  // prevent clipping
  strokeWidth: 4,
  trailWidth: 1,
  easing: 'easeInOut',
  duration: 1400,
  text: {
    autoStyleContainer: false
  },
  from: { color: '#aaa', width: 1 },
  to: { color: '#333', width: 4 },
  // Set default step function for all animate calls
  step: function(state, circle) {
    circle.path.setAttribute('stroke', state.color);
    circle.path.setAttribute('stroke-width', state.width);

    var value = Math.round(circle.value() * 100);
    if (value === 0) {
      circle.setText('');
    } else {
      circle.setText(value);
    }

  }
});
bar.text.style.fontFamily = '"Raleway", Helvetica, sans-serif';
bar.text.style.fontSize = '2rem';

bar.animate(1.0);  // Number from 0.0 to 1.0