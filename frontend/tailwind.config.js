/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: {
          bg: "#F7F8FA",
        },
        surface: {
          card: "#FFFFFF",
        },
        sidebar: {
          bg: "#14243A",
          hover: "#1C3250",
          active: "#233E63",
        },
        primary: {
          blue: "#315B85",
          hover: "#26496B",
          light: "#EBF2F8",
        },
        border: {
          subtle: "#E2E8F0",
          medium: "#CBD5E0",
        },
        text: {
          primary: "#1A202C",
          secondary: "#4A5568",
          muted: "#718096",
        },
        risk: {
          high: {
            base: "#C53030",
            text: "#9B2C2C",
            surface: "#FFF5F5",
            border: "#FEB2B2",
          },
          mod: {
            base: "#D69E2E",
            text: "#B7791F",
            surface: "#FFFAF0",
            border: "#FBD38D",
          },
          low: {
            base: "#2F855A",
            text: "#22543D",
            surface: "#F0FFF4",
            border: "#C6F6D5",
          },
          suppressed: {
            base: "#4A5568",
            text: "#2D3748",
            surface: "#EDF2F7",
            border: "#CBD5E0",
          },
          ggg: {
            base: "#5A67D8",
            text: "#434190",
            surface: "#EBF4FF",
            border: "#C3DAFE",
          },
        },
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
      },
    },
  },
  plugins: [],
}
