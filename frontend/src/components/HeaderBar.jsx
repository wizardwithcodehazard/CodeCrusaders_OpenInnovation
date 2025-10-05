import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";

const HeaderBar = () => {
  return (
    <header className="fixed top-0 left-0 w-full z-50 bg-gradient-to-b from-slate-900/95 via-slate-900/80 to-transparent backdrop-blur-md border-b border-white/10 shadow-lg shadow-black/30">
      <div className="max-w-7xl mx-auto px-6 lg:px-12 py-4 flex items-center justify-between">
        
        {/* Logo + Name */}
        <Link to="/" className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl overflow-hidden bg-white/10 border border-white/10 flex items-center justify-center">
            <img 
              src="/logo.png" 
              alt="Logo" 
              className="w-full h-full object-contain"
            />
          </div>
          <span className="text-xl font-bold text-white tracking-wide">
            Wolftor
          </span>
        </Link>

        {/* Right Side - Redirect to Chat Page */}
        <Link
          to="/chat"
          className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-white/10 border border-white/10 hover:bg-white/20 text-slate-200 hover:text-white transition-all backdrop-blur-md shadow-md shadow-black/20"
        >
          
          <span className="font-medium text-sm">Go to Chat</span>
        </Link>
      </div>
    </header>
  );
};

export default HeaderBar;
