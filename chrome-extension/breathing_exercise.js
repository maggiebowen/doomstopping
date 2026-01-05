const sequence = [
    // Interactive Intro (waitForClick: true)
    { text: "You asked me to remind you to take a break when you feel distressed.", waitForClick: true },
    { text: "Your facial expressions and physiological signals indicate you are distressed.", waitForClick: true },
    { text: "Let's do a breathing exercise together to relax.", waitForClick: true },

    // Meditation (Auto flow)
    { text: "Breathing in, I know I am breathing in.", duration: 6000 },
    { text: "Breathing out, I know I am breathing out.", duration: 6000 },
    { text: "In…", duration: 5000 },
    { text: "Out…", duration: 5000 },

    { text: "Breathing in, I become aware of my body.", duration: 5000 },
    { text: "Breathing out, I release any tension.", duration: 5000 },
    { text: "Aware of body…", duration: 5000 },
    { text: "Releasing tension…", duration: 5000 },

    { text: "Breathing in, I arrive in this present moment.", duration: 6000 },
    { text: "Breathing out, I feel at home in the here and now.", duration: 6000 },
    { text: "Arriving…", duration: 5000 },
    { text: "Home…", duration: 5000 },

    { text: "Breathing in, I calm my mind.", duration: 6000 },
    { text: "Breathing out, I let go of all worries.", duration: 6000 },
    { text: "Calm…", duration: 5000 },
    { text: "Letting go…", duration: 5000 }
];

let textElement;
let hintElement;
let currentIndex = 0;
let isWaitingForClick = false;

function showNext() {
    if (currentIndex >= sequence.length) {
        // End of sequence. Fade out text.
        if (textElement) textElement.classList.remove('visible');
        if (hintElement) hintElement.classList.remove('visible-hint');
        return;
    }

    const item = sequence[currentIndex];

    // Fade Out
    if (textElement) textElement.classList.remove('visible');
    if (hintElement) hintElement.classList.remove('visible-hint');

    // Wait/Swap
    setTimeout(() => {
        if (!textElement) return;

        textElement.innerText = item.text;
        textElement.classList.add('visible');

        // Play bell sound on odd indices starting from 3 (Inhale prompts)
        // Index 3: "Breathing in..." (Sound)
        // Index 4: "Breathing out..." (No Sound)
        // Index 5: "In..." (Sound)
        if (currentIndex >= 3 && currentIndex % 2 !== 0) {
            const bell = document.getElementById('bellSound');
            if (bell) {
                bell.currentTime = 0;
                bell.play().catch(e => console.log("Audio play failed:", e));
            }
        }

        if (item.waitForClick) {
            isWaitingForClick = true;
            if (hintElement) hintElement.classList.add('visible-hint'); // Show hint
            // Logic handles click listener below
        } else {
            isWaitingForClick = false;
            // Schedule next auto
            currentIndex++;
            setTimeout(showNext, item.duration);
        }

    }, 500);
}

document.addEventListener('DOMContentLoaded', () => {
    textElement = document.getElementById('mainText');
    hintElement = document.getElementById('clickHint');

    // Click Logic
    document.body.addEventListener('click', () => {
        if (isWaitingForClick) {
            // User clicked, proceed
            isWaitingForClick = false;
            currentIndex++;
            showNext();
        }
    });

    // Start delay
    setTimeout(showNext, 1000);
});
