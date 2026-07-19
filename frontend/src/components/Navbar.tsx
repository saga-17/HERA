import { Link, useLocation } from "react-router-dom";

const navItems = [
  { path: "/", label: "Home" },
  { path: "/inference", label: "Inference" },
  { path: "/about", label: "About" },
];

export default function Navbar() {
  const location = useLocation();

  return (
    <nav className="border-b border-slate-700/50 bg-hera-surface/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-hera-primary to-hera-secondary flex items-center justify-center font-bold text-sm">
              H
            </div>
            <span className="font-semibold text-lg">HERA-VLM</span>
          </Link>

          <div className="flex gap-1">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  location.pathname === item.path
                    ? "bg-hera-primary/20 text-indigo-300"
                    : "text-slate-400 hover:text-white hover:bg-slate-700/50"
                }`}
              >
                {item.label}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}
