/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        hera: {
          primary: "#6366f1",
          secondary: "#8b5cf6",
          dark: "#0f172a",
          surface: "#1e293b",
        },
      },
    },
  },
  plugins: [],
};
