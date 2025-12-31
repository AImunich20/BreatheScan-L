export default {
  content: [
    "./templates/hom.html",    // scan Flask templates
  ],
  theme: { extend: {} },
  plugins: [],
  safelist: [
    "bg-white", "text-blue-700", "px-6", "py-3", "rounded-full", "font-semibold", "shadow-lg",
    "hover:bg-blue-50", "text-indigo-900", "text-indigo-800", "text-xl", "text-5xl",
    "font-bold", "leading-tight", "mb-6", "mb-8", "grid", "md:grid-cols-2", "gap-12",
    "items-center", "flex", "min-h-screen"
  ]
}
