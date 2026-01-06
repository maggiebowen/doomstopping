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

    // cut for demo time
    // { text: "Breathing in, I become aware of my body.", duration: 5000 },
    // { text: "Breathing out, I release any tension.", duration: 5000 },
    // { text: "Aware of body…", duration: 5000 },
    // { text: "Releasing tension…", duration: 5000 },

    { text: "Breathing in, I arrive in this present moment.", duration: 6000 },
    { text: "Breathing out, I feel at home in the here and now.", duration: 6000 },
    { text: "Arriving…", duration: 5000 },
    { text: "Home…", duration: 5000 },

    { text: "Breathing in, I calm my mind.", duration: 6000 },
    { text: "Breathing out, I let go of all worries.", duration: 6000 },
    { text: "Calm…", duration: 5000 },
    { text: "Letting go…", duration: 5000 },

    // Final redirection
    {
        text: "Instead of continuing to doomscroll, read about how you can consume media more mindfully, in a way that aligns with your values and reduces your physiological stress",
        waitForClick: true,
        isFinal: true
    }
];

let textElement;
let hintElement;
let currentIndex = 0;
let isWaitingForClick = false;

function showNext() {
    if (currentIndex >= sequence.length) {
        // This shouldn't be reached if logic works right, but just in case
        return;
    }

    const item = sequence[currentIndex];

    // Fade Out
    if (textElement) textElement.classList.remove('visible');
    if (hintElement) hintElement.classList.remove('visible-hint');

    // Wait/Swap
    setTimeout(() => {
        if (!textElement) return;

        // Play bell sound IMMEDIATELY (odd indices >= 3, but not the final one if we don't want it)
        // Checks: index 3, 5, 7, ... and NOT final
        if (currentIndex >= 3 && currentIndex % 2 !== 0 && !item.isFinal) {
            const bell = document.getElementById('bellSound');
            if (bell) {
                bell.currentTime = 0;
                bell.play().catch(e => console.log("Audio play failed:", e));
            }
        }

        textElement.innerText = item.text;
        textElement.classList.add('visible');

        if (item.waitForClick) {
            isWaitingForClick = true;
            if (hintElement) {
                hintElement.innerText = item.isFinal ? "(Click to exit to mindful consumption)" : "(Click to continue)";
                hintElement.classList.add('visible-hint');
            }
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
            const item = sequence[currentIndex];
            if (item && item.isFinal) {
                // Redirect logic
                window.location.href = "https://plumvillage.app/the-practice-of-mindful-consumption/";
                return;
            }

            // User clicked, proceed
            isWaitingForClick = false;
            currentIndex++;
            showNext();
        }
    });

    // Start delay
    setTimeout(showNext, 1000);
});
