import { useState, useRef, useCallback, useEffect } from "react";
import {
  Send,
  Loader2,
  X,
  Download,
  Sparkles,
  Paperclip,
} from "lucide-react";
import NavBar from "./Navbar";

const ChatPage = () => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [problemText, setProblemText] = useState("");
  const [mathematicianMode, setMathematicianMode] = useState(false);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [generatingVideo, setGeneratingVideo] = useState(null);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);
  const messagesEndRef = useRef(null);

  const API_BASE_URL = "http://localhost:8000";

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleFileUpload = useCallback((file) => {
    if (
      file &&
      (file.type.startsWith("image/") || file.type === "application/pdf")
    ) {
      setUploadedFile(file);
    } else {
      alert("Please upload a valid image file (JPG, PNG) or PDF");
    }
  }, []);

  const handleFileInputChange = (e) => {
    const file = e.target.files[0];
    handleFileUpload(file);
  };

  const removeFile = () => {
    setUploadedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const sendMessage = async () => {
    if (isLoading) return;

    let userMessage = "";
    let messageType = "";

    if (uploadedFile) {
      userMessage = `Uploaded image: ${uploadedFile.name}`;
      messageType = "image";
    } else if (problemText.trim()) {
      userMessage = problemText.trim();
      messageType = "text";
    } else {
      return;
    }

    const newUserMessage = {
      id: Date.now(),
      type: "user",
      content: userMessage,
      messageType: messageType,
      file: uploadedFile,
    };

    setMessages((prev) => [...prev, newUserMessage]);
    setIsLoading(true);

    try {
      let response;
      if (messageType === "image") {
        // Handle image upload
        const formData = new FormData();
        formData.append("image", uploadedFile);

        response = await fetch(`${API_BASE_URL}/api/image-to-text/`, {
          method: "POST",
          body: formData,
        });

        // Safe parse
        const ct = response.headers.get("content-type") || "";
        const data = ct.includes("application/json")
          ? await response.json()
          : { text: await response.text() };

        const extractedText = data.text;

        if (extractedText && extractedText.trim().length > 0) {
          // Now explain the problem with extracted text
          const explainResponse = await fetch(
            `${API_BASE_URL}/api/explain-problem/`,
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({ problem: extractedText }),
            }
          );

          const explainCT = explainResponse.headers.get("content-type") || "";
          const explainData = explainCT.includes("application/json")
            ? await explainResponse.json()
            : { explanation: await explainResponse.text() };

          const aiMessage = {
            id: Date.now() + 1,
            type: "ai",
            content: explainData,
            extractedText: extractedText,
            tutorMode: mathematicianMode, // preserve tutor mode flag
          };

          setMessages((prev) => [...prev, aiMessage]);
        } else {
          const aiMessage = {
            id: Date.now() + 1,
            type: "ai",
            content:
              "I couldn't extract any text from the image. Please try uploading a clearer image or type your problem directly.",
            extractedText: null,
            tutorMode: mathematicianMode,
          };
          setMessages((prev) => [...prev, aiMessage]);
        }
      } else {
        // Handle text problem: route to chat-simple when mathematicianMode is false (default),
        // otherwise route to explain-problem when mathematicianMode is true.
        const endpoint = mathematicianMode
          ? "/api/explain-problem/"
          : "/api/chat-simple/";

        // IMPORTANT: chat-simple expects key 'message' or 'problem' — we send 'message'
        response = await fetch(`${API_BASE_URL}${endpoint}`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ message: problemText }),
        });

        const ct = response.headers.get("content-type") || "";
        const data = ct.includes("application/json")
          ? await response.json()
          : { response: await response.text() };

        const aiMessage = {
          id: Date.now() + 1,
          type: "ai",
          content: data,
          tutorMode: mathematicianMode,
        };

        setMessages((prev) => [...prev, aiMessage]);
      }
    } catch (error) {
      console.error("Error:", error);
      const errorMessage = {
        id: Date.now() + 1,
        type: "ai",
        content: `Sorry, I encountered an error: ${error.message}. Please try again.`,
        tutorMode: mathematicianMode,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setProblemText("");
      setUploadedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const generateVideo = async (messageId, problemText) => {
    console.log("🎬 Starting video generation for message:", messageId);
    setGeneratingVideo(messageId);

    try {
      const response = await fetch(`${API_BASE_URL}/api/generate-video/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ problem: problemText }),
      });

      const ct = response.headers.get("content-type") || "";
      const data = ct.includes("application/json")
        ? await response.json()
        : { error: await response.text() };

      console.log("✅ Video generation response received:", data);

      // Update the message with video data
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === messageId ? { ...msg, videoData: data } : msg
        )
      );
    } catch (error) {
      console.error("❌ Video generation error:", error);

      // Add error message
      const errorMessage = {
        id: Date.now(),
        type: "ai",
        content: `Sorry, I couldn't generate a video: ${error.message}. Please try again.`,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setGeneratingVideo(null);
    }
  };

  // Markdown / explanation formatting
  const formatExplanation = (explanation) => {
    if (!explanation) return "No explanation available";

    const lines = String(explanation).split("\n");
    const formattedLines = [];
    let inCodeBlock = false;
    let codeBlockContent = [];
    let codeBlockLanguage = "";

    lines.forEach((line, index) => {
      // Code block start/end
      if (line.startsWith("```")) {
        if (!inCodeBlock) {
          inCodeBlock = true;
          codeBlockLanguage = line.replace("```", "").trim();
          codeBlockContent = [];
        } else {
          inCodeBlock = false;
          formattedLines.push(
            <div key={`code-${index}`} className="my-4">
              <div className="bg-slate-950/80 rounded-lg border border-white/10 overflow-hidden">
                {codeBlockLanguage && (
                  <div className="bg-white/5 px-4 py-2 text-xs text-slate-400 font-mono border-b border-white/10">
                    {codeBlockLanguage}
                  </div>
                )}
                <pre className="p-4 overflow-x-auto">
                  <code className="text-sm text-slate-200 font-mono">
                    {codeBlockContent.join("\n")}
                  </code>
                </pre>
              </div>
            </div>
          );
          codeBlockContent = [];
          codeBlockLanguage = "";
        }
        return;
      }

      if (inCodeBlock) {
        codeBlockContent.push(line);
        return;
      }

      // #### header -> big highlighted block (strip asterisks)
      if (line.startsWith("#### ")) {
        let text = line.replace("#### ", "").trim();
        text = text.replace(/^\*+|\*+$/g, "");
        formattedLines.push(
          <div key={index} className="my-4">
            <h4 className="text-slate-100 font-bold text-lg bg-white/5 px-4 py-2.5 rounded-lg border-l-4 border-blue-400/50">
              {text}
            </h4>
          </div>
        );
        return;
      }

      // ### header -> subtitle (strip asterisks), no star icons
      if (line.startsWith("### ")) {
        let text = line.replace("### ", "").trim();
        text = text.replace(/^\*+|\*+$/g, "");
        formattedLines.push(
          <div key={index} className="my-3">
            <h3 className="text-blue-300 font-semibold text-base">{text}</h3>
          </div>
        );
        return;
      }

      // inline code using backticks
      if (line.includes("`") && !line.startsWith("```")) {
        const parts = line.split(/(`[^`]+`)/g);
        formattedLines.push(
          <div key={index} className="mb-2">
            {parts.map((part, partIndex) => {
              if (part.startsWith("`") && part.endsWith("`")) {
                const code = part.slice(1, -1);
                return (
                  <code
                    key={partIndex}
                    className="bg-slate-950/20 text-white px-1.5 py-0.5 rounded text-sm font-mono"
                  >
                    {code}
                  </code>
                );
              }
              return <span key={partIndex}>{part}</span>;
            })}
          </div>
        );
        return;
      }

      // bold **text**
      if (line.includes("**")) {
        const parts = line.split(/(\*\*.*?\*\*)/g);
        formattedLines.push(
          <div key={index} className="mb-2">
            {parts.map((part, partIndex) => {
              if (part.startsWith("**") && part.endsWith("**")) {
                const boldText = part.slice(2, -2);
                return (
                  <strong key={partIndex} className="font-bold text-slate-100">
                    {boldText}
                  </strong>
                );
              }
              return <span key={partIndex}>{part}</span>;
            })}
          </div>
        );
        return;
      }

      // empty line
      if (line.trim() === "") {
        formattedLines.push(<div key={index} className="mb-2"></div>);
        return;
      }

      // normal text
      formattedLines.push(
        <div key={index} className="mb-2 text-slate-300 leading-relaxed">
          {line}
        </div>
      );
    });

    return (
      <div className="bg-white/5 backdrop-blur-sm rounded-xl p-5 border border-white/10">
        <div className="space-y-2">{formattedLines}</div>
      </div>
    );
  };

  const formatResults = (data) => {
    if (!data) return "No data received";

    // If API returned { explanation: "..." }
    if (data.explanation) {
      return formatExplanation(data.explanation);
    }
    // If API returned { response: "..." }
    if (data.response) {
      return formatExplanation(data.response);
    }

    // If it's a plain object
    if (typeof data === "object") {
      return Object.entries(data).map(([key, value]) => (
        <div key={key} className="mb-2">
          <strong className="text-blue-300">{key}:</strong>{" "}
          <span className="text-slate-300">{String(value)}</span>
        </div>
      ));
    }

    // Otherwise show text
    return <div className="text-slate-300">{String(data)}</div>;
  };

  const ThinkingAnimation = () => (
    <div className="flex justify-start mb-6 animate-fadeIn">
      <div className="max-w-3xl p-6 rounded-2xl bg-white/5 backdrop-blur-md border border-white/10">
        <div className="flex items-center space-x-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-full bg-blue-500/20 flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-blue-400 animate-pulse" />
            </div>
            <div className="absolute inset-0 rounded-full bg-blue-400/30 animate-ping"></div>
          </div>
          <div className="flex space-x-2">
            <div
              className="w-2.5 h-2.5 bg-blue-400 rounded-full animate-bounce"
              style={{ animationDelay: "0ms" }}
            ></div>
            <div
              className="w-2.5 h-2.5 bg-blue-400 rounded-full animate-bounce"
              style={{ animationDelay: "150ms" }}
            ></div>
            <div
              className="w-2.5 h-2.5 bg-blue-400 rounded-full animate-bounce"
              style={{ animationDelay: "300ms" }}
            ></div>
          </div>
          <span className="text-slate-200 font-medium">
            Analyzing your problem...
          </span>
        </div>
      </div>
    </div>
  );

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Toggle UI component
  const TutorToggle = ({ checked, onChange }) => {
    return (
      <button
        onClick={() => onChange(!checked)}
        className={`flex items-center gap-3 px-3 py-2 rounded-full text-sm font-medium transition-all focus:outline-none ${
          checked
            ? "bg-gradient-to-br from-blue-500 to-blue-400 shadow-[0_6px_20px_rgba(59,130,246,0.28)] text-white border border-blue-300/30"
            : "bg-white/5 text-slate-300 border border-white/10"
        }`}
        title="Tutor mode (video explanation)"
      >
        <div
          className={`w-5 h-5 rounded-full flex items-center justify-center transition-all ${
            checked ? "bg-white/10" : "bg-transparent"
          }`}
        >
          <Sparkles className="w-3 h-3" />
        </div>
        <span>{checked ? "Tutor mode" : "Tutor mode"}</span>
      </button>
    );
  };

  return (
    <div className="min-h-screen w-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 relative overflow-hidden">
      <div className="absolute inset-0 opacity-30 bg-gradient-to-br from-blue-500/5 to-purple-500/5"></div>

      <NavBar />

      <div className="relative flex flex-col h-screen max-w-10xl mx-auto">
        {/* Chat Messages Area */}
        <div className="flex-1 overflow-y-auto px-6 lg:px-72 pt-32">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center space-y-4 max-w-2xl">
                <div className="w-20 h-20 mx-auto bg-gradient-to-br from-blue-500/20 to-purple-500/20 backdrop-blur-md rounded-2xl flex items-center justify-center border border-white/10">
                  <Sparkles className="w-10 h-10 text-blue-400" />
                </div>
                <h1 className="text-4xl font-bold text-white">
                  How can I help you today?
                </h1>
                <p className="text-slate-400">
                  Upload an image or type your problem to get started
                </p>
              </div>
            </div>
          ) : (
            <div className="space-y-6 pb-6">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${
                    message.type === "user" ? "justify-end" : "justify-start"
                  } animate-fadeIn`}
                >
                  <div
                    className={`max-w-3xl p-5 rounded-2xl ${
                      message.type === "user"
                        ? "bg-gradient-to-br from-blue-600 to-blue-500 text-white shadow-lg shadow-blue-500/20"
                        : "bg-white/5 backdrop-blur-md text-slate-100 border border-white/10"
                    }`}
                  >
                    {message.type === "user" &&
                      message.messageType === "image" && (
                        <div className="mb-3">
                          <img
                            src={URL.createObjectURL(message.file)}
                            alt="Uploaded"
                            className="max-w-xs rounded-lg border-2 border-white/20"
                          />
                        </div>
                      )}

                    <div className="whitespace-pre-wrap">
                      {message.type === "ai"
                        ? formatResults(message.content)
                        : message.content}
                    </div>

                    {message.type === "ai" &&
                      message.content &&
                      !message.videoData &&
                      message.tutorMode && (
                        <div className="mt-4 pt-4 border-t border-white/10">
                          <button
                            onClick={() =>
                              generateVideo(
                                message.id,
                                message.content.problem ||
                                  message.extractedText ||
                                  message.content.response ||
                                  ""
                              )
                            }
                            disabled={generatingVideo === message.id}
                            className="bg-white/10 backdrop-blur-md text-white px-5 py-2.5 rounded-lg hover:bg-white/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2 font-medium border border-white/20"
                          >
                            {generatingVideo === message.id ? (
                              <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                <span>Generating Video...</span>
                              </>
                            ) : (
                              <>
                                <Download className="w-4 h-4" />
                                <span>Explain with Video</span>
                              </>
                            )}
                          </button>
                        </div>
                      )}

                    {message.videoData && (
                      <div className="mt-4 pt-4 border-t border-white/10">
                        <h4 className="text-blue-300 font-semibold mb-3 flex items-center">
                          <Sparkles className="w-4 h-4 mr-2" />
                          Video Explanation:
                        </h4>
                        <video
                          controls
                          className="w-[50vw]  rounded-lg border border-white/20"
                        >
                          <source
                            src={`${API_BASE_URL}${message.videoData.video_url}`}
                            type="video/mp4"
                          />
                          Your browser does not support the video tag.
                        </video>
                        {message.videoData.transcript && (
                          <div className="mt-3 bg-white/5 rounded-lg p-4 border border-white/10">
                            <h5 className="text-blue-300 font-semibold mb-2">
                              Transcript:
                            </h5>
                            <p className="text-sm text-slate-300 whitespace-pre-wrap leading-relaxed">
                              {message.videoData.transcript}
                            </p>
                          </div>
                        )}
                      </div>
                    )}

                    {message.extractedText && (
                      <div className="mt-3 pt-3 border-t border-white/10">
                        <p className="text-sm text-slate-400">
                          <strong className="text-blue-300">
                            Extracted text:
                          </strong>{" "}
                          {message.extractedText}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {isLoading && <ThinkingAnimation />}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Fixed Input Area */}
        <div className="sticky bottom-0 px-6 py-6 bg-gradient-to-t from-slate-900 via-slate-900/95 to-transparent">
          <div className="max-w-3xl mx-auto">
            {/* File Preview */}
            {uploadedFile && (
              <div className="mb-3 bg-white/5 backdrop-blur-md rounded-xl p-3 border border-white/10 animate-fadeIn">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-12 h-12 rounded-lg overflow-hidden bg-white/10">
                      <img
                        src={URL.createObjectURL(uploadedFile)}
                        alt="Preview"
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div>
                      <p className="text-slate-200 font-medium text-sm">
                        {uploadedFile.name}
                      </p>
                      <p className="text-slate-400 text-xs">
                        {(uploadedFile.size / 1024).toFixed(2)} KB
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={removeFile}
                    className="text-slate-400 hover:text-red-400 transition-colors p-2 hover:bg-red-400/10 rounded-lg"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}

            {/* Input Box */}
            <div className="bg-white/10 backdrop-blur-md rounded-2xl border border-white/20 shadow-2xl shadow-black/20 overflow-hidden">
              <div className="flex items-end p-2 gap-2 relative">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex-shrink-0 p-3 rounded-xl bg-white/10 hover:bg-white/20 text-slate-300 hover:text-white transition-all border border-white/10 hover:border-white/30"
                  title="Attach Image"
                >
                  <Paperclip className="w-5 h-5" />
                </button>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*,.pdf"
                  onChange={handleFileInputChange}
                  className="hidden"
                />

                <div className="absolute left-16 bottom-3">
                  <TutorToggle
                    checked={mathematicianMode}
                    onChange={(v) => setMathematicianMode(v)}
                  />
                </div>

                <textarea
                  ref={textareaRef}
                  value={problemText}
                  onChange={(e) => {
                    setProblemText(e.target.value);
                    e.target.style.height = "auto";
                    e.target.style.height =
                      Math.min(e.target.scrollHeight, 128) + "px";
                  }}
                  onKeyPress={handleKeyPress}
                  placeholder="Message Wolftor..."
                  className="flex-1 bg-transparent border-none text-slate-100 placeholder-slate-400 resize-none focus:outline-none px-3 pl-36 py-3 max-h-32"
                  rows={1}
                  style={{ minHeight: "44px" }}
                />

                <button
                  onClick={sendMessage}
                  disabled={isLoading || (!uploadedFile && !problemText.trim())}
                  className="flex-shrink-0 p-3 rounded-xl bg-gradient-to-br from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white transition-all disabled:opacity-40 disabled:cursor-not-allowed shadow-lg shadow-blue-500/20 border border-white/20"
                >
                  {isLoading ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Send className="w-5 h-5" />
                  )}
                </button>
              </div>
            </div>

            <p className="text-center text-slate-500 text-xs mt-3">
              Wolftor can make mistakes. Check important info.
            </p>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-fadeIn {
          animation: fadeIn 0.5s ease-out;
        }

        ::-webkit-scrollbar {
          width: 8px;
        }

        ::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.03);
          border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.12);
          border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.18);
        }
      `}</style>
    </div>
  );
};

export default ChatPage;
