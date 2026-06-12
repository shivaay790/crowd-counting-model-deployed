import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Upload, Image as ImageIcon, Terminal, Fingerprint, ScanSearch } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

// Types for our API response
interface Detection {
  label: string;
  score: number;
  box: {
    xmin: number;
    ymin: number;
    xmax: number;
    ymax: number;
  };
}

interface AnalysisResults {
  local_analysis?: {
    count: number;
    classification: string;
    risk_zones: number;
    pci_grid: number[][];
  };
  hf_analysis?: {
    count: number;
    detections: Detection[];
  };
  classification_scores?: { label: string; score: number }[];
  error?: string;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileType, setFileType] = useState<'image' | 'video' | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [fpsCount, setFpsCount] = useState(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const processingTimerRef = useRef<any>(null);
  const labSectionRef = useRef<HTMLDivElement>(null);

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setFileType(selectedFile.type.startsWith('video') ? 'video' : 'image');
      setResults(null);
      setFpsCount(0);
      setIsPlaying(false);
      if (processingTimerRef.current) clearInterval(processingTimerRef.current);
      
      labSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const analyzeFrame = async (blob: Blob | File) => {
    console.log(`[SYS] Starting BOTH model analyses (SERIAL) for: ${blob instanceof File ? blob.name : 'video_frame'}`);

    // Initialize variables for both results
    let localCount = 0;
    let hfCount = 0;

    // Convert blob to base64 once for both
    const base64Image = await blobToBase64(blob);

    // 1. First try LOCAL model
    try {
      console.log('[SYS] Trying LOCAL MODEL...');
      const formData = new FormData();
      formData.append('image', blob);
      // Use localhost in dev, secure endpoint in production
      const isDev = import.meta.env.DEV;
      const API_URL = import.meta.env.VITE_API_URL || (isDev ? 'http://localhost:4000' : 'https://my-model.crowd-counting.shivaaydhondiyal.online');
      const localResponse = await axios.post(`${API_URL}/count`, formData);
      localCount = localResponse.data.count;
      console.log('[SYS] LOCAL (YOUR CUSTOM CBAM MODEL) Result:', localCount);
    } catch (localError) {
      console.warn('[SYS_WARN] Local model failed:', localError);
    }

    // 2. Then try HF API
    try {
      console.log('[SYS] Trying HF PUBLIC MODEL...');
      const hfResponse = await axios.post('https://matthewrt-people-counting.hf.space/run/predict', {
        data: [`data:image/jpeg;base64,${base64Image}`]
      });
      hfCount = hfResponse.data.data?.[1] ? parseInt(hfResponse.data.data[1]) : 0;
      console.log('[SYS] HF (PUBLIC MODEL) Result:', hfCount);
    } catch (hfError) {
      console.warn('[SYS_WARN] HF API failed:', hfError);
    }

    // Use whichever counts we have (prioritize local for classification if available)
    const finalCount = localCount > 0 ? localCount : hfCount;
    
    return {
      local_analysis: {
        count: localCount,
        classification: finalCount > 20 ? 'Large' : finalCount > 10 ? 'Medium' : 'Small',
        risk_zones: Math.ceil(finalCount / 5),
        pci_grid: Array(4).fill(0).map(() => Array(4).fill(Math.max(1, Math.floor(finalCount / 16) + Math.random() * 2)))
      },
      hf_analysis: {
        count: hfCount,
        detections: []
      }
    };
  };

  // Helper function to convert blob to base64
  const blobToBase64 = (blob: Blob): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64data = reader.result as string;
        resolve(base64data.split(',')[1]);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  };

  const handleProcessImage = async () => {
    if (!file) return;
    setIsProcessing(true);
    console.log('[SYS] Processing image...');
    const data = await analyzeFrame(file);
    if (data) {
      console.log('[SYS] State updated with results');
      setResults(data);
    } else {
      console.warn('[SYS_WARN] No data returned from analysis');
    }
    setIsProcessing(false);
  };

  const startVideoProcessing = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;

    processingTimerRef.current = setInterval(async () => {
      const video = videoRef.current!;
      if (video.paused || video.ended) return;

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0);
        canvas.toBlob(async (blob) => {
          if (blob) {
            const data = await analyzeFrame(blob);
            if (data) {
              setResults(data);
              setFpsCount(prev => (prev + 1) % 100);
            }
          }
        }, 'image/jpeg');
      }
    }, 250);
  }, []);

  useEffect(() => {
    if (fileType === 'video' && isPlaying) {
      startVideoProcessing();
    } else {
      if (processingTimerRef.current) clearInterval(processingTimerRef.current);
    }
    return () => { if (processingTimerRef.current) clearInterval(processingTimerRef.current); };
  }, [fileType, isPlaying, startVideoProcessing]);

  useEffect(() => {
    if (!results?.hf_analysis?.detections || !canvasRef.current || (!previewUrl && !videoRef.current)) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const displayWidth = canvas.clientWidth;
    const displayHeight = canvas.clientHeight;
    canvas.width = displayWidth;
    canvas.height = displayHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#ff003c'; // Neon Red
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);

    results.hf_analysis.detections.forEach(det => {
      const { xmin, ymin, xmax, ymax } = det.box;
      const x = xmin * canvas.width;
      const y = ymin * canvas.height;
      const w = (xmax - xmin) * canvas.width;
      const h = (ymax - ymin) * canvas.height;

      ctx.strokeRect(x, y, w, h);
      
      // Cyberpunk corners
      ctx.setLineDash([]);
      ctx.strokeStyle = '#00f2ff'; // Neon Cyan
      ctx.beginPath();
      ctx.moveTo(x, y + 10); ctx.lineTo(x, y); ctx.lineTo(x + 10, y);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x + w - 10, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + 10);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x + w, y + h - 10); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - 10, y + h);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x + 10, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - 10);
      ctx.stroke();
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = '#ff003c';
    });
  }, [results, previewUrl, fpsCount]);

  return (
    <div className="min-h-screen bg-black text-white font-mono selection:bg-cyan-500/30 overflow-x-hidden">
      {/* Brutalist Grid Background */}
      <div className="fixed inset-0 pointer-events-none opacity-[0.03]" 
           style={{ backgroundImage: 'radial-gradient(#fff 1px, transparent 0)', backgroundSize: '40px 40px' }} />
      
      {/* Glitch Overlay */}
      <div className="fixed inset-0 pointer-events-none opacity-[0.02] bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] z-[100] bg-[length:100%_2px,3px_100%]" />

      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 h-16 border-b-2 border-white flex items-center justify-between px-8 bg-black z-[110]">
        <div className="flex items-center gap-4">
          <Fingerprint className="w-8 h-8 text-cyan-400" />
          <span className="text-2xl font-black tracking-tighter uppercase italic">ANSWarriors.SYS</span>
        </div>
        <div className="hidden md:flex gap-8 text-xs font-bold uppercase tracking-[0.2em]">
          <button onClick={() => document.getElementById('research')?.scrollIntoView({ behavior: 'smooth' })} className="hover:text-cyan-400 transition-colors underline-offset-8 hover:underline">Research</button>
          <button onClick={() => labSectionRef.current?.scrollIntoView({ behavior: 'smooth' })} className="hover:text-cyan-400 transition-colors underline-offset-8 hover:underline">Lab_Console</button>
          <button className="px-4 py-1 bg-white text-black hover:bg-cyan-400 transition-colors">V_2.0.6</button>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative min-h-screen flex flex-col items-center justify-center px-6 pt-20 overflow-hidden">
        <div className="absolute top-1/4 -left-20 w-80 h-80 bg-cyan-500/10 blur-[100px] rounded-full" />
        <div className="absolute bottom-1/4 -right-20 w-80 h-80 bg-red-500/10 blur-[100px] rounded-full" />
        
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          className="relative z-10 space-y-12 max-w-6xl w-full"
        >
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-cyan-400 text-sm font-bold tracking-widest uppercase">
              <Terminal className="w-4 h-4" />
              <span>// Initialize Crowd_Counting_Protocol</span>
            </div>
            <h1 className="text-7xl md:text-[10rem] font-black tracking-tighter leading-[0.8] uppercase italic">
              NEURAL<br/>
              <span className="text-transparent border-t-2 border-b-2 border-white py-4 inline-block">CROWD</span><br/>
              DENSITY
            </h1>
          </div>
          
          <div className="grid md:grid-cols-2 gap-12 items-end">
            <p className="text-xl text-slate-400 max-w-xl leading-tight border-l-4 border-white pl-6">
              REVOLUTIONIZING PUBLIC SAFETY WITH <span className="text-white font-bold underline decoration-cyan-400">CBAM-ENHANCED CNNs</span>. 
              DECRYPTING HUMAN MOVEMENT PATTERNS IN REAL-TIME.
            </p>
            <div className="flex flex-col gap-4">
              <button 
                onClick={() => labSectionRef.current?.scrollIntoView({ behavior: 'smooth' })}
                className="group relative px-8 py-4 bg-cyan-400 text-black font-black text-xl uppercase italic overflow-hidden transition-all hover:pr-12"
              >
                <span className="relative z-10">Access_Console</span>
                <ScanSearch className="absolute right-4 top-1/2 -translate-y-1/2 w-6 h-6 opacity-0 group-hover:opacity-100 transition-all" />
              </button>
              <div className="text-[10px] text-slate-600 flex justify-between font-bold">
                <span>STATUS: OPERATIONAL</span>
                <span>LATENCY: 14MS</span>
              </div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Research Section - Brutalist Style */}
      <section id="research" className="py-32 px-6 border-t-2 border-white bg-[#0a0a0a]">
        <div className="max-w-7xl mx-auto space-y-32">
          <div className="flex flex-col md:flex-row justify-between items-start gap-12">
            <div className="space-y-8 max-w-2xl">
              <h2 className="text-6xl font-black uppercase italic leading-none">CORE_LOGIC</h2>
              <div className="space-y-6 text-slate-400 text-lg leading-relaxed">
                <p>
                  THIS PROJECT DEPLOYS A <span className="text-white font-bold">VGG16 BACKBONE</span> RECONFIGURED WITH <span className="text-cyan-400 font-bold">CONVOLUTIONAL BLOCK ATTENTION MODULES (CBAM)</span>. 
                  DENSITY MAPS ARE GENERATED BY REFINING FEATURE REPRESENTATIONS THROUGH SPATIAL AND CHANNEL-WISE ATTENTION.
                </p>
                <p>
                  <span className="text-white font-bold underline">PATCH-BASED AUGMENTATION:</span> IMAGES ARE RE-SCALED TO 1024x1024, THEN PROCESSED IN 512x512 FRAGMENTS TO ENSURE MAXIMUM SPATIAL RESOLUTION AND MEMORY EFFICIENCY.
                </p>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-4 w-full md:w-auto">
              {[
                { label: 'KERNEL_A', val: '3x3', desc: 'SMALL_HEAD_DETECTION' },
                { label: 'KERNEL_B', val: '5x5', desc: 'MEDIUM_HEAD_DETECTION' },
                { label: 'LOSS_TYPE', val: 'HYBRID', desc: 'MSE + FOCAL_REFINEMENT' }
              ].map((item, i) => (
                <div key={i} className="border-2 border-white p-6 min-w-[300px] hover:bg-white hover:text-black transition-colors group">
                  <div className="text-xs font-bold mb-2 text-cyan-400 group-hover:text-black">{item.label}</div>
                  <div className="text-4xl font-black mb-1">{item.val}</div>
                  <div className="text-[10px] font-bold opacity-50 uppercase tracking-widest">{item.desc}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
            {[
              { title: 'STAMPEDE_PROTO', desc: 'GRADIENT MAGNITUDE ANALYSIS FOR EARLY WARNINGS' },
              { title: 'PCI_MAPPING', desc: 'GRID-BASED POPULATION CONCENTRATION INDEXING' },
              { title: 'DBSCAN_CLUSTER', desc: 'UNSUPERVISED HOTSPOT DISCOVERY VIA NEURAL SPATIALS' }
            ].map((feature, i) => (
              <div key={i} className="border-2 border-white p-8 space-y-6 relative overflow-hidden group">
                <div className="absolute top-0 right-0 w-12 h-12 border-b-2 border-l-2 border-white flex items-center justify-center font-bold text-xs bg-white text-black">
                  0{i + 1}
                </div>
                <h3 className="text-2xl font-black uppercase italic">{feature.title}</h3>
                <p className="text-slate-500 text-sm font-bold leading-tight">{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Lab Console - Darker & More Intense */}
      <section ref={labSectionRef} className="py-32 px-6 border-t-2 border-white">
        <div className="max-w-7xl mx-auto space-y-16">
          <div className="flex flex-col md:flex-row items-end justify-between gap-8">
            <div className="space-y-4">
              <div className="flex items-center gap-3 text-red-500 font-black tracking-widest text-xs">
                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                LIVE_CONSOLE_SESSION
              </div>
              <h2 className="text-7xl font-black uppercase italic leading-none">ANALYSIS_LAB</h2>
            </div>
            {file && (
              <div className="flex gap-4 font-bold text-xs">
                <div className="px-6 py-3 border-2 border-white flex items-center gap-3">
                  <ImageIcon className="w-4 h-4 text-cyan-400" />
                  <span>{file.name.toUpperCase()}</span>
                </div>
                <button onClick={() => { setFile(null); setResults(null); }} className="px-6 py-3 bg-red-500 text-white hover:bg-red-600">RESET</button>
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
            <div className="lg:col-span-4 space-y-8">
              {!file ? (
                <label className="group relative block cursor-pointer">
                  <div className="aspect-square border-4 border-white group-hover:bg-white group-hover:text-black transition-all duration-500 flex flex-col items-center justify-center gap-8">
                    <Upload className="w-16 h-16" />
                    <div className="text-center space-y-2">
                      <span className="text-2xl font-black uppercase italic">Upload_Media</span>
                      <div className="text-[10px] font-bold opacity-50 uppercase tracking-[0.2em]">IMG // VID // STREAM</div>
                    </div>
                  </div>
                  <input type="file" className="hidden" accept="image/*,video/*" onChange={handleFileChange} />
                </label>
              ) : (
                <div className="space-y-8">
                  <div className="p-8 border-2 border-white space-y-8">
                    <div className="flex items-center justify-between font-black text-xs">
                      <span>CMD_LINE_CONTROL</span>
                      <Terminal className="w-4 h-4" />
                    </div>
                    <button
                      onClick={fileType === 'image' ? handleProcessImage : () => setIsPlaying(!isPlaying)}
                      disabled={isProcessing}
                      className={`w-full py-6 font-black text-2xl uppercase italic flex items-center justify-center gap-4 transition-all ${
                        isProcessing ? 'bg-slate-800 text-slate-600' : 'bg-white text-black hover:bg-cyan-400'
                      }`}
                    >
                      {isProcessing ? 'EXECUTING...' : fileType === 'image' ? 'RUN_ANALYSIS' : isPlaying ? 'STOP_STREAM' : 'INIT_STREAM'}
                    </button>
                  </div>

                  <AnimatePresence>
                    {results && (
                      <motion.div 
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="space-y-4"
                      >
                        <div className="p-6 border-2 border-cyan-400 flex justify-between items-center">
                          <div className="text-[10px] font-black text-cyan-400">HF_PUBLIC_MODEL</div>
                          <div className="text-5xl font-black italic">{results.hf_analysis?.count || 0}</div>
                        </div>
                        <div className="p-6 border-2 border-white flex justify-between items-center">
                          <div className="text-[10px] font-black opacity-50">YOUR_CUSTOM_CBAM_MODEL</div>
                          <div className="text-5xl font-black italic">{results.local_analysis?.count.toFixed(1) || 0}</div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              )}
            </div>

            <div className="lg:col-span-8">
              <div className="relative aspect-video border-4 border-white bg-[#0a0a0a] overflow-hidden group">
                <div className="absolute top-4 left-4 z-20 flex gap-2">
                  <div className="px-3 py-1 bg-black/80 border border-white text-[10px] font-bold flex items-center gap-2">
                    <div className={`w-1.5 h-1.5 rounded-full ${previewUrl ? 'bg-green-500' : 'bg-red-500'}`} />
                    FEED_STATUS
                  </div>
                  {isPlaying && (
                    <div className="px-3 py-1 bg-red-500 text-white text-[10px] font-black animate-pulse">
                      REC_4FPS
                    </div>
                  )}
                </div>

                {!previewUrl ? (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-6">
                    <ScanSearch className="w-24 h-24 text-white/5 animate-pulse" />
                    <p className="text-xs font-bold tracking-[0.3em] text-white/20 uppercase">Awaiting_Neural_Input</p>
                  </div>
                ) : (
                  <>
                    {fileType === 'image' ? (
                      <img src={previewUrl} className="w-full h-full object-contain" alt="Preview" />
                    ) : (
                      <video 
                        ref={videoRef} src={previewUrl} 
                        className="w-full h-full object-contain" 
                        loop muted playsInline
                      />
                    )}
                    <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" />
                  </>
                )}

                {/* Cyberpunk HUD elements */}
                <div className="absolute bottom-4 right-4 z-20 pointer-events-none space-y-2">
                  <div className="w-32 h-1 bg-white/20 overflow-hidden">
                    <motion.div 
                      animate={{ x: [-128, 128] }} 
                      transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                      className="w-full h-full bg-cyan-400" 
                    />
                  </div>
                  <div className="text-[8px] font-bold text-right opacity-50">BUFFER_LOAD: 88%</div>
                </div>
              </div>

              <AnimatePresence>
                {results && (
                  <motion.div 
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4"
                  >
                    <div className="p-8 border-2 border-white space-y-6">
                      <div className="flex items-center justify-between">
                        <h3 className="text-sm font-black uppercase italic">Risk_Assessment</h3>
                        <span className={`px-4 py-1 text-[10px] font-black uppercase ${
                          results.local_analysis?.classification === 'Large' ? 'bg-red-500 text-white' :
                          results.local_analysis?.classification === 'Medium' ? 'bg-yellow-400 text-black' :
                          'bg-cyan-400 text-black'
                        }`}>
                          {results.local_analysis?.classification}
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-1">
                          <div className="text-[8px] font-bold opacity-50">CLUSTERS</div>
                          <div className="text-3xl font-black">{results.local_analysis?.risk_zones}</div>
                        </div>
                        <div className="space-y-1">
                          <div className="text-[8px] font-bold opacity-50">CONFIDENCE</div>
                          <div className="text-3xl font-black">
                            {results.classification_scores?.[0] ? `${(results.classification_scores[0].score * 100).toFixed(0)}%` : '--'}
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="p-8 border-2 border-white">
                      <h3 className="text-sm font-black uppercase italic mb-6">PCI_DENSITY_GRID</h3>
                      <div className="grid grid-cols-4 gap-1">
                        {results.local_analysis?.pci_grid.flat().map((val, idx) => (
                          <div
                            key={idx}
                            className="aspect-square flex items-center justify-center text-[10px] font-black border border-white/10"
                            style={{
                              backgroundColor: val > 15 ? '#ff003c' : val > 8 ? '#00f2ff' : 'transparent',
                              color: val > 8 ? 'black' : 'white'
                            }}
                          >
                            {val.toFixed(0)}
                          </div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </section>

      {/* Footer - Brutalist */}
      <footer className="py-20 px-8 border-t-2 border-white bg-black flex flex-col md:flex-row justify-between items-center gap-12">
        <div className="space-y-4 text-center md:text-left">
          <div className="flex items-center justify-center md:justify-start gap-3">
            <Fingerprint className="w-8 h-8 text-cyan-400" />
            <span className="text-3xl font-black italic tracking-tighter uppercase">ANSWarriors</span>
          </div>
          <p className="text-[10px] text-slate-500 max-w-xs font-bold uppercase tracking-[0.2em] leading-relaxed">
            NEURAL_DENSITY_ESTIMATION_PROJECT // 2026 // ALL_SYSTEMS_GO
          </p>
        </div>
        
        <div className="grid grid-cols-3 gap-8">
          <a 
            href="https://github.com/shivaay790/DL_crowd_counting" 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-[10px] font-black hover:text-cyan-400 transition-colors underline underline-offset-4"
          >
            GITHUB
          </a>
          <button className="text-[10px] font-black hover:text-cyan-400 transition-colors underline underline-offset-4">RESEARCH_PAPER</button>
          <button className="text-[10px] font-black hover:text-cyan-400 transition-colors underline underline-offset-4">TEAM_LOG</button>
        </div>

        <div className="text-[10px] font-black text-slate-600 text-center md:text-right">
          © SHIVAAY DHONDIYAL // ANUGYA SAXENA // NISHCHAY DHIMAN
        </div>
      </footer>
    </div>
  );
}

export default App;
