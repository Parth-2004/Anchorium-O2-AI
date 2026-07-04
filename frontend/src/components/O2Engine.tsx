import React, { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Upload, Send, CheckCircle, Shield } from 'lucide-react';
import { FadeUp } from './FadeUp';

type AppState = 'hero' | 'dashboard';

export default function O2Engine() {
  const [appState, setAppState] = useState<AppState>('hero');
  const [uploaded, setUploaded] = useState(false);

  const headingWords = "OMNI ENGINE: CROSS-BORDER CREDIT ARBITRAGE, SOLVED.".split(" ");

  return (
    <div className="font-helvetica text-[#F1E5AC] min-h-screen relative overflow-hidden bg-[#0B0B0C]">
      <style>{`
        @import url('https://db.onlinewebfonts.com/c/e66905e07608167a84e6ad52f638c3c6?family=Helvetica+Now+Var');
        
        .font-helvetica {
          font-family: 'Helvetica Now Var', sans-serif;
        }
        
        /* Custom scrollbar for chat */
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(11, 11, 12, 0.5);
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(212, 175, 55, 0.2);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(212, 175, 55, 0.4);
        }
        
        /* Hide scrollbar for quick actions */
        .hide-scrollbar::-webkit-scrollbar {
          display: none;
        }
        .hide-scrollbar {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }
      `}</style>

      {/* PERMANENT BACKGROUND VIDEO */}
      <video
        autoPlay
        muted
        loop
        playsInline
        className="fixed top-0 left-0 w-full h-[100vh] object-cover -z-10"
      >
        <source src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260514_135830_bb6491d1-9b66-4aec-9722-13b4dfe3fb46.mp4" type="video/mp4" />
      </video>
      <div className="fixed inset-0 bg-[#0B0B0C]/75 z-0 pointer-events-none" />

      {/* STATE MANAGER */}
      <div className="relative z-10 w-full h-screen">
        <AnimatePresence mode="wait">
          {appState === 'hero' ? (
            <motion.div
              key="hero"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, filter: 'blur(10px)', scale: 0.95 }}
              transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
              className="h-full flex flex-col justify-center px-[18px] md:px-[32px] pt-[90px] md:pt-[70px] pb-[32px]"
            >
              <div className="max-w-[760px] flex flex-col items-start w-full mx-auto lg:mx-0 lg:ml-[10vw]">
                {/* HEADING */}
                <h2 className="flex flex-wrap gap-[0.25em] m-0 p-0 leading-[1.1]">
                  {headingWords.map((word, i) => (
                    <FadeUp key={i} delay={0.15 + i * 0.08} duration={0.7} as="span">
                      <span 
                        className="text-[clamp(26px,3.5vw,48px)] font-bold uppercase text-[#D4AF37] block"
                        style={{ textShadow: '0 4px 24px rgba(212, 175, 55, 0.2)' }}
                      >
                        {word}
                      </span>
                    </FadeUp>
                  ))}
                </h2>

                {/* SUBTEXT */}
                <FadeUp delay={0.9} y={24} duration={0.8}>
                  <p className="text-[#F1E5AC] opacity-90 max-w-[480px] mt-6 text-[16px] md:text-[18px] leading-relaxed">
                    Automating SBLC-backed INR liquidity, real-time FEMA compliance, and zero-knowledge underwriting for international founders expanding to India.
                  </p>
                </FadeUp>

                {/* CTA BUTTON */}
                <FadeUp delay={1.1} duration={0.8}>
                  <motion.button
                    onClick={() => setAppState('dashboard')}
                    whileHover={{ scale: 1.05, boxShadow: '0 0 30px rgba(212, 175, 55, 0.4)' }}
                    whileTap={{ scale: 0.98 }}
                    className="mt-10 bg-[#D4AF37] text-[#0B0B0C] uppercase font-bold px-8 py-4 border border-[#F1E5AC] shadow-[0_0_20px_rgba(212,175,55,0.15)] tracking-wider rounded-sm transition-colors cursor-pointer"
                  >
                    GET STARTED: INITIALIZE OMNI ENGINE
                  </motion.button>
                </FadeUp>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0, filter: 'blur(10px)', y: 20 }}
              animate={{ opacity: 1, filter: 'blur(0px)', y: 0 }}
              exit={{ opacity: 0, filter: 'blur(10px)', y: -20 }}
              transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
              className="h-full w-full flex flex-col md:flex-row p-4 md:p-8 gap-6 box-border pt-16 md:pt-8"
            >
              {/* THE VAULT (Left Panel) */}
              <div className="flex-1 flex flex-col backdrop-blur-md bg-[#0B0B0C]/40 border border-[#D4AF37]/30 rounded-2xl p-6 md:p-8 relative overflow-hidden shadow-[0_0_40px_rgba(0,0,0,0.8)]">
                {/* Vault Header */}
                <div className="flex items-center gap-3 mb-8">
                  <Shield className="text-[#D4AF37] w-6 h-6" />
                  <h3 className="text-[#D4AF37] uppercase tracking-widest font-semibold text-sm md:text-base">The Vault</h3>
                </div>
                
                {/* Dropzone */}
                <div 
                  className={`flex-1 border-2 border-dashed rounded-xl flex flex-col items-center justify-center p-6 text-center transition-all duration-300 ${uploaded ? 'border-green-500/40 bg-green-500/5' : 'border-[#D4AF37]/40 hover:bg-[#D4AF37]/5 cursor-pointer group'}`}
                  onClick={() => !uploaded && setUploaded(true)}
                >
                  {uploaded ? (
                    <motion.div 
                      initial={{ scale: 0.8, opacity: 0 }} 
                      animate={{ scale: 1, opacity: 1 }} 
                      className="flex flex-col items-center"
                    >
                      <CheckCircle className="text-green-400 w-16 h-16 mb-4 drop-shadow-[0_0_15px_rgba(74,222,128,0.4)]" />
                      <p className="text-green-400 font-bold text-lg md:text-xl mb-2">Document Embedded Locally</p>
                      <p className="text-[#F1E5AC]/70 text-sm">100% Privacy Preserved</p>
                    </motion.div>
                  ) : (
                    <>
                      <div className="w-20 h-20 rounded-full bg-[#0B0B0C] border border-[#D4AF37]/20 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-500">
                        <Upload className="text-[#D4AF37] w-8 h-8 group-hover:text-[#F1E5AC] transition-colors" />
                      </div>
                      <p className="text-[#F1E5AC] font-medium text-lg mb-2 group-hover:text-[#D4AF37] transition-colors">Drag & Drop Financial Documents</p>
                      <p className="text-[#F1E5AC]/60 text-sm mb-8">Supports PDF, XLSX, GAAP Reports</p>
                      <button className="px-6 py-3 bg-[#D4AF37]/10 text-[#D4AF37] border border-[#D4AF37]/30 rounded uppercase text-xs font-bold tracking-wider group-hover:bg-[#D4AF37]/20 transition-colors">
                        Browse Files
                      </button>
                    </>
                  )}
                </div>
              </div>

              {/* OMNI-CHAT (Right Panel) */}
              <div className="flex-[1.5] flex flex-col backdrop-blur-md bg-[#0B0B0C]/40 border border-[#D4AF37]/30 rounded-2xl relative overflow-hidden shadow-[0_0_40px_rgba(0,0,0,0.8)] h-full">
                {/* Chat Header */}
                <div className="p-5 border-b border-[#D4AF37]/20 flex items-center justify-between bg-black/40">
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]" />
                    <h3 className="text-[#D4AF37] uppercase tracking-widest font-semibold text-sm md:text-base">Omni-Chat</h3>
                  </div>
                  <span className="text-[10px] md:text-xs text-[#F1E5AC]/60 border border-[#F1E5AC]/20 px-3 py-1.5 rounded uppercase tracking-wider bg-black/20">
                    Agent: Compliance Copilot
                  </span>
                </div>

                {/* Chat History */}
                <div className="flex-1 overflow-y-auto p-5 md:p-6 flex flex-col gap-6 custom-scrollbar">
                  {/* AI Message */}
                  <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }} className="flex justify-start">
                    <div className="bg-[#0B0B0C] border border-[#D4AF37]/40 p-4 md:p-5 rounded-2xl rounded-tl-sm max-w-[90%] md:max-w-[80%] shadow-[0_4px_16px_rgba(0,0,0,0.4)]">
                      <p className="text-[#F1E5AC] text-sm leading-relaxed">
                        Omni Engine initialized. Secure enclave active. How can I assist you with cross-border structuring or compliance today?
                      </p>
                    </div>
                  </motion.div>
                  
                  {/* User Message */}
                  <motion.div initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 1.2 }} className="flex justify-end">
                    <div className="bg-[#D4AF37] text-[#0B0B0C] p-4 md:p-5 rounded-2xl rounded-tr-sm max-w-[90%] md:max-w-[80%] shadow-[0_4px_16px_rgba(212,175,55,0.15)]">
                      <p className="font-semibold text-sm leading-relaxed">
                        What are the current FEMA regulations for repatriating SaaS revenue?
                      </p>
                    </div>
                  </motion.div>
                  
                  {/* AI Message */}
                  <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 2.5 }} className="flex justify-start">
                    <div className="bg-[#0B0B0C] border border-[#D4AF37]/40 p-4 md:p-5 rounded-2xl rounded-tl-sm max-w-[90%] md:max-w-[80%] shadow-[0_4px_16px_rgba(0,0,0,0.4)]">
                      <p className="text-[#F1E5AC] text-sm leading-relaxed">
                        Based on FEMA 120/RB-2004, SaaS revenue repatriation falls under current account transactions. You must ensure OPGSP (Online Payment Gateway Service Provider) limits are respected, currently capped at $10,000 per transaction for export of services. Would you like me to draft a compliance checklist?
                      </p>
                    </div>
                  </motion.div>
                </div>

                {/* Quick Actions */}
                <div className="px-5 md:px-6 pb-3 pt-2 flex gap-2 overflow-x-auto hide-scrollbar border-t border-[#D4AF37]/10 bg-[#0B0B0C]/40">
                  {['Convert US GAAP to IndAS', 'Run FEMA Check', 'Analyze SBLC Terms'].map((action, i) => (
                    <motion.button 
                      key={i} 
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className="whitespace-nowrap px-4 py-2 rounded-full border border-[#D4AF37]/30 bg-[#0B0B0C]/80 text-[#D4AF37] text-[10px] md:text-xs uppercase tracking-wider hover:bg-[#D4AF37] hover:text-[#0B0B0C] transition-colors shadow-sm"
                    >
                      {action}
                    </motion.button>
                  ))}
                </div>

                {/* Input */}
                <div className="p-5 md:p-6 pt-2 bg-black/40">
                  <div className="relative flex items-center bg-[#0B0B0C] border border-[#D4AF37]/40 rounded-xl overflow-hidden focus-within:border-[#D4AF37] focus-within:shadow-[0_0_15px_rgba(212,175,55,0.2)] transition-all duration-300">
                    <input 
                      type="text" 
                      placeholder="Query the Omni Engine..." 
                      className="w-full bg-transparent text-[#F1E5AC] p-4 pr-14 outline-none placeholder-[#F1E5AC]/30 font-light text-sm"
                    />
                    <button className="absolute right-2 p-2.5 bg-[#D4AF37]/10 text-[#D4AF37] hover:bg-[#D4AF37] hover:text-[#0B0B0C] rounded-lg transition-colors">
                      <Send className="w-4 h-4 md:w-5 md:h-5" />
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
