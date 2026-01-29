/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"IBM Plex Sans"', 'system-ui', '-apple-system', 'sans-serif'],
      },
      colors: {
        // Primary teal palette
        primary: {
          50: '#EFFAF9',
          100: '#D5F2EF',
          200: '#ABE5DF',
          300: '#6DD4CA',
          400: '#38BFB1',
          500: '#0F6E66',
          600: '#0D5F58',
          700: '#0A4E48',
          800: '#083D39',
          900: '#062E2B',
        },
        // Neutral palette
        neutral: {
          50: '#F7FAFB',
          100: '#F0F4F5',
          200: '#E2E8EA',
          300: '#C8D1D4',
          400: '#9AA8AD',
          500: '#6B7D84',
          600: '#4A5B61',
          700: '#374349',
          800: '#252F33',
          900: '#0F172A',
        },
        // Surface colors
        surface: {
          DEFAULT: '#FFFFFF',
          elevated: '#FFFFFF',
          sunken: '#F7FAFB',
        },
        // Semantic colors
        success: {
          50: '#EDFCF0',
          100: '#D3F9DB',
          500: '#137F2E',
          600: '#0F6B26',
          700: '#0B571E',
        },
        warning: {
          50: '#FEF9EC',
          100: '#FDF0CD',
          500: '#B45309',
          600: '#9A4708',
          700: '#7F3A06',
        },
        error: {
          50: '#FEF2F2',
          100: '#FEE2E2',
          500: '#C62828',
          600: '#B71C1C',
          700: '#991B1B',
        },
        // Accent colors for conditions
        healthy: '#137F2E',
        diabetes: '#B45309',
        hypertension: '#C62828',
        'heart-disease': '#7B1FA2',
      },
      borderRadius: {
        DEFAULT: '6px',
        sm: '4px',
        md: '6px',
        lg: '8px',
        xl: '12px',
      },
      boxShadow: {
        'subtle': '0 1px 2px 0 rgb(0 0 0 / 0.04)',
        'card': '0 1px 3px 0 rgb(0 0 0 / 0.06), 0 1px 2px -1px rgb(0 0 0 / 0.06)',
        'dropdown': '0 4px 6px -1px rgb(0 0 0 / 0.08), 0 2px 4px -2px rgb(0 0 0 / 0.04)',
        'modal': '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
      },
      fontSize: {
        'xs': ['0.75rem', { lineHeight: '1rem' }],
        'sm': ['0.875rem', { lineHeight: '1.25rem' }],
        'base': ['1rem', { lineHeight: '1.5rem' }],
        'lg': ['1.125rem', { lineHeight: '1.75rem' }],
        'xl': ['1.25rem', { lineHeight: '1.75rem' }],
        '2xl': ['1.5rem', { lineHeight: '2rem' }],
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem',
      },
      transitionDuration: {
        '150': '150ms',
        '200': '200ms',
      },
    },
  },
  plugins: [],
};
