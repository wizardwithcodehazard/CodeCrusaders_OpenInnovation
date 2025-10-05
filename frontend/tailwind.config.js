/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        'background-dark': '#1A1A3A',
        'background-light': '#2A2A5A',
        'primary-orange': '#FF7F00',
        'secondary-orange': '#CD7F32',
        'text-light': '#FFFFFF',
        'text-muted': '#E0E0E0',
        'card-background': '#2C2C4F',
        'pro-tip-background': '#1A1A3A',
      },
    },
  },
  plugins: [],
}
