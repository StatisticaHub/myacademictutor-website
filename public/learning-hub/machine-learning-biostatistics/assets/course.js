document.addEventListener("DOMContentLoaded", () => {
  const copyButtons = document.querySelectorAll("[data-copy-target]");

  copyButtons.forEach((button) => {
    button.addEventListener("click", async () => {
      const targetId = button.getAttribute("data-copy-target");
      const target = document.getElementById(targetId);

      if (!target) return;

      try {
        await navigator.clipboard.writeText(target.innerText);
        const originalText = button.innerText;
        button.innerText = "Copied";
        setTimeout(() => {
          button.innerText = originalText;
        }, 1500);
      } catch (error) {
        button.innerText = "Copy failed";
      }
    });
  });

  const quizButtons = document.querySelectorAll("[data-answer]");

  quizButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const parent = button.closest(".checkpoint");
      const feedback = parent.querySelector(".feedback-box");
      const isCorrect = button.getAttribute("data-answer") === "correct";

      parent.querySelectorAll(".option-btn").forEach((btn) => {
        btn.classList.remove("correct", "incorrect");
      });

      button.classList.add(isCorrect ? "correct" : "incorrect");

      if (feedback) {
        feedback.classList.remove("correct", "incorrect", "show");
        feedback.classList.add(isCorrect ? "correct" : "incorrect", "show");
        feedback.innerHTML = isCorrect
          ? feedback.getAttribute("data-correct")
          : feedback.getAttribute("data-incorrect");
      }
    });
  });

  const flashcards = document.querySelectorAll(".flashcard");

  flashcards.forEach((card) => {
    card.addEventListener("click", () => {
      card.classList.toggle("open");
    });
  });

  const sliders = document.querySelectorAll("[data-slider-output]");

  sliders.forEach((slider) => {
    const outputId = slider.getAttribute("data-slider-output");
    const output = document.getElementById(outputId);

    const updateOutput = () => {
      if (!output) return;

      const value = Number(slider.value);
      output.innerText = `Threshold = ${value.toFixed(2)}`;

      if (value < 0.4) {
        output.innerText += " · More sensitive, more false positives";
      } else if (value > 0.7) {
        output.innerText += " · More specific, more false negatives";
      } else {
        output.innerText += " · Balanced starting point";
      }
    };

    slider.addEventListener("input", updateOutput);
    updateOutput();
  });
});
