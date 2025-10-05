import { Link } from 'react-router-dom';
import { Zap, Brain, Camera, Calculator, ArrowRight, Sparkles, CheckCircle } from 'lucide-react';
import HeaderBar from './HeaderBar';


const LandingPage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 opacity-30 bg-gradient-to-br from-blue-500/5 to-purple-500/5"></div>
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-600/10 via-transparent to-transparent"></div>

    <HeaderBar/>

      {/* Hero Section */}
      <main className="relative z-10 container mx-auto px-6 py-16 pt-32">
        <div className="text-center mb-20">
          <div className="inline-flex items-center px-4 py-2 bg-white/5 backdrop-blur-md rounded-full border border-white/10 mb-6">
            <Sparkles className="w-4 h-4 text-blue-400 mr-2" />
            <span className="text-sm text-slate-300">AI-Powered Problem Solving</span>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
            Solve Problems
            <span className="block bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Instantly with AI
            </span>
          </h1>
          
          <p className="text-xl text-slate-300 mb-10 max-w-3xl mx-auto leading-relaxed">
            Upload images or type problems - our advanced AI combines computer vision, 
            natural language processing, and mathematical computation to solve any challenge.
          </p>
          
          <Link 
            to="/chat"
            className="inline-flex items-center px-8 py-4 bg-gradient-to-r from-blue-600 to-blue-500 text-white font-semibold rounded-xl hover:from-blue-500 hover:to-blue-400 transition-all duration-300 shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/40 hover:scale-105 transform"
          >
            Get Started Free
            <ArrowRight className="ml-2 w-5 h-5" />
          </Link>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-6 mb-20 max-w-6xl mx-auto">
          <div className="group bg-white/5 backdrop-blur-md p-8 rounded-2xl border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all duration-300 hover:scale-105 transform hover:shadow-xl">
            <div className="flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-xl mb-6 group-hover:scale-110 transition-transform">
              <Camera className="w-8 h-8 text-blue-400" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-4">Image Recognition</h3>
            <p className="text-slate-400 leading-relaxed">
              Upload photos of handwritten notes, printed problems, or diagrams. 
              Our AI extracts text and understands visual content.
            </p>
          </div>

          <div className="group bg-white/5 backdrop-blur-md p-8 rounded-2xl border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all duration-300 hover:scale-105 transform hover:shadow-xl">
            <div className="flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-xl mb-6 group-hover:scale-110 transition-transform">
              <Brain className="w-8 h-8 text-purple-400" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-4">AI Analysis</h3>
            <p className="text-slate-400 leading-relaxed">
              Advanced language models understand context, identify problem types, 
              and generate appropriate solution strategies.
            </p>
          </div>

          <div className="group bg-white/5 backdrop-blur-md p-8 rounded-2xl border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all duration-300 hover:scale-105 transform hover:shadow-xl">
            <div className="flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-xl mb-6 group-hover:scale-110 transition-transform">
              <Calculator className="w-8 h-8 text-blue-400" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-4">Mathematical Engine</h3>
            <p className="text-slate-400 leading-relaxed">
              Powered by Wolfram Engine for precise calculations, symbolic math, 
              and step-by-step solutions across all domains.
            </p>
          </div>
        </div>

        {/* How It Works */}
        <div className="bg-white/5 backdrop-blur-md p-10 rounded-3xl border border-white/10 mb-20 max-w-6xl mx-auto hover:bg-white/10 transition-all duration-300">
          <h2 className="text-4xl font-bold text-white mb-12 text-center">How It Works</h2>
          <div className="grid md:grid-cols-4 gap-8">
            <div className="text-center group">
              <div className="flex items-center justify-center w-14 h-14 bg-gradient-to-br from-blue-600 to-blue-500 rounded-full mx-auto mb-4 shadow-lg shadow-blue-500/30 group-hover:scale-110 transition-transform">
                <span className="text-white font-bold text-lg">1</span>
              </div>
              <h3 className="font-semibold text-white mb-2 text-lg">Upload or Type</h3>
              <p className="text-sm text-slate-400">Share your problem via image or text input</p>
            </div>
            <div className="text-center group">
              <div className="flex items-center justify-center w-14 h-14 bg-gradient-to-br from-blue-600 to-blue-500 rounded-full mx-auto mb-4 shadow-lg shadow-blue-500/30 group-hover:scale-110 transition-transform">
                <span className="text-white font-bold text-lg">2</span>
              </div>
              <h3 className="font-semibold text-white mb-2 text-lg">AI Processing</h3>
              <p className="text-sm text-slate-400">Our AI analyzes and understands the problem</p>
            </div>
            <div className="text-center group">
              <div className="flex items-center justify-center w-14 h-14 bg-gradient-to-br from-blue-600 to-blue-500 rounded-full mx-auto mb-4 shadow-lg shadow-blue-500/30 group-hover:scale-110 transition-transform">
                <span className="text-white font-bold text-lg">3</span>
              </div>
              <h3 className="font-semibold text-white mb-2 text-lg">Solution Generation</h3>
              <p className="text-sm text-slate-400">Advanced algorithms compute the solution</p>
            </div>
            <div className="text-center group">
              <div className="flex items-center justify-center w-14 h-14 bg-gradient-to-br from-blue-600 to-blue-500 rounded-full mx-auto mb-4 shadow-lg shadow-blue-500/30 group-hover:scale-110 transition-transform">
                <span className="text-white font-bold text-lg">4</span>
              </div>
              <h3 className="font-semibold text-white mb-2 text-lg">Get Results</h3>
              <p className="text-sm text-slate-400">Receive detailed explanations and answers</p>
            </div>
          </div>
        </div>

        {/* Benefits Section */}
        <div className="grid md:grid-cols-2 gap-6 mb-20 max-w-6xl mx-auto">
          <div className="bg-white/5 backdrop-blur-md p-6 rounded-xl border border-white/10 hover:bg-white/10 transition-all">
            <div className="flex items-start space-x-3">
              <CheckCircle className="w-6 h-6 text-blue-400 mt-1 flex-shrink-0" />
              <div>
                <h4 className="text-white font-semibold mb-1">Step-by-Step Solutions</h4>
                <p className="text-slate-400 text-sm">Get detailed explanations for every problem</p>
              </div>
            </div>
          </div>
          <div className="bg-white/5 backdrop-blur-md p-6 rounded-xl border border-white/10 hover:bg-white/10 transition-all">
            <div className="flex items-start space-x-3">
              <CheckCircle className="w-6 h-6 text-blue-400 mt-1 flex-shrink-0" />
              <div>
                <h4 className="text-white font-semibold mb-1">Multiple Subjects</h4>
                <p className="text-slate-400 text-sm">Math, Physics, Chemistry, and more</p>
              </div>
            </div>
          </div>
          <div className="bg-white/5 backdrop-blur-md p-6 rounded-xl border border-white/10 hover:bg-white/10 transition-all">
            <div className="flex items-start space-x-3">
              <CheckCircle className="w-6 h-6 text-blue-400 mt-1 flex-shrink-0" />
              <div>
                <h4 className="text-white font-semibold mb-1">24/7 Availability</h4>
                <p className="text-slate-400 text-sm">Get help anytime, anywhere</p>
              </div>
            </div>
          </div>
          <div className="bg-white/5 backdrop-blur-md p-6 rounded-xl border border-white/10 hover:bg-white/10 transition-all">
            <div className="flex items-start space-x-3">
              <CheckCircle className="w-6 h-6 text-blue-400 mt-1 flex-shrink-0" />
              <div>
                <h4 className="text-white font-semibold mb-1">Video Explanations</h4>
                <p className="text-slate-400 text-sm">Watch AI-generated solution videos</p>
              </div>
            </div>
          </div>
        </div>

        {/* CTA Section */}
        <div className="text-center bg-gradient-to-br from-blue-600/10 to-purple-600/10 backdrop-blur-md p-12 rounded-3xl border border-white/10 max-w-4xl mx-auto">
          <div className="flex items-center justify-center mb-6">
            <Sparkles className="w-10 h-10 text-blue-400 mr-3" />
            <h2 className="text-4xl font-bold text-white">Ready to Solve Problems?</h2>
          </div>
          <p className="text-slate-300 mb-8 max-w-2xl mx-auto text-lg leading-relaxed">
            Join thousands of students, professionals, and problem-solvers who trust Wolftor 
            for accurate, step-by-step solutions.
          </p>
          <Link 
            to="/chat"
            className="inline-flex items-center px-10 py-4 bg-gradient-to-r from-blue-600 to-blue-500 text-white font-bold rounded-xl hover:from-blue-500 hover:to-blue-400 transition-all duration-300 shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/40 text-lg hover:scale-105 transform"
          >
            Start Solving Now
            <ArrowRight className="ml-2 w-6 h-6" />
          </Link>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/10 mt-20 backdrop-blur-sm bg-slate-900/50">
        <div className="container mx-auto px-6 py-8">
          <div className="flex items-center justify-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-500 rounded-lg flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <span className="text-slate-400">© 2024 Wolftor - AI-Powered Problem Solver</span>
          </div>
        </div>
      </footer>

      <style>{`
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-10px); }
        }
      `}</style>
    </div>
  );
};

export default LandingPage;