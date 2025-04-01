import React, { useState, useEffect } from "react";
import { Camera, CameraOff } from "lucide-react";
import { cn } from "@/lib/utils";
import { FaceDetection } from "@/utils/types";
import { getVideoStreamUrl } from "@/utils/api";

interface VideoFeedProps {
  className?: string;
  detectedFaces?: FaceDetection[];
  isActive?: boolean;
  backendUrl?: string;
}

const VideoFeed: React.FC<VideoFeedProps> = ({
  className,
  detectedFaces = [],
  isActive = true,
  backendUrl,
}) => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isStreamingMode, setIsStreamingMode] = useState(false);

  useEffect(() => {
    // Simulate loading time for UI feedback
    const timer = setTimeout(() => {
      setIsLoaded(true);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    // Set streaming mode if backend URL is provided
    setIsStreamingMode(Boolean(backendUrl) && isActive);
  }, [backendUrl, isActive]);

  return (
    <div
      className={cn(
        "video-container overflow-hidden relative rounded-xl bg-black/5 transition-all duration-500",
        isLoaded ? "animate-scale-in opacity-100" : "opacity-0",
        className
      )}
    >
      {isActive ? (
        <div className="aspect-video w-full relative flex items-center justify-center">
          {isStreamingMode && backendUrl ? (
            // Real video stream from backend
            <img
              src={getVideoStreamUrl(backendUrl)}
              alt="Video Stream"
              className="w-full h-full object-cover"
              style={{
                backgroundColor: "black",
                transform: "scaleX(-1)", // Mirror the video
              }}
              onLoad={() => setIsLoaded(true)}
              onError={(e) => {
                console.error("Video stream error:", e);
                setIsLoaded(false);
              }}
            />
          ) : (
            // Fallback to placeholder for demo mode
            <>
              <div className="absolute inset-0 bg-gradient-to-tr from-slate-100 to-blue-50 animate-pulse-gentle opacity-40" />
              <div className="px-4 py-2 rounded-lg bg-white/80 backdrop-blur-sm shadow-sm border border-white/20 text-sm text-slate-600">
                {backendUrl
                  ? "Connecting to video stream..."
                  : "Demo mode active. Connect to backend for live video."}
              </div>
            </>
          )}

          {/* Face detection boxes overlay */}
          <div className="absolute inset-0 z-10">
            {isLoaded &&
              detectedFaces.map((face) => (
                <div
                  key={face.id}
                  className="face-box absolute"
                  style={{
                    top: `${face.y}%`,
                    left: `${face.x}%`,
                    width: `${face.width}%`,
                    height: `${face.height}%`,
                    border: "2px solid",
                    borderColor:
                      face.name !== "Visitor"
                        ? "rgba(0, 119, 255, 0.8)"
                        : "rgba(255, 90, 0, 0.8)",
                  }}
                >
                  <div className="absolute -top-6 left-0 px-2 py-0.5 rounded-md text-xs text-white bg-blue-500 shadow-sm whitespace-nowrap">
                    {face.name} {face.confidence > 0.8 && "✓"}
                  </div>
                </div>
              ))}
          </div>

          {/* Recording indicator */}
          {isLoaded && isActive && (
            <div className="absolute top-3 right-3 flex items-center space-x-1.5 px-2 py-1 rounded-full bg-black/20 backdrop-blur-md text-white text-xs">
              <div className="h-2 w-2 rounded-full bg-red-500 recording-indicator"></div>
              <span>Recording</span>
            </div>
          )}
        </div>
      ) : (
        <div className="aspect-video w-full flex items-center justify-center bg-slate-100">
          <div className="flex flex-col items-center text-slate-500">
            <CameraOff className="h-10 w-10 mb-2 text-slate-400" />
            <p className="text-sm">Camera is offline</p>
            <button className="mt-4 px-3 py-1.5 rounded-md bg-blue-500 text-white text-sm flex items-center">
              <Camera className="h-4 w-4 mr-1.5" />
              Enable Camera
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoFeed;
