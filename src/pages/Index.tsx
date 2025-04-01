import React, { useState, useEffect, useCallback } from "react";
import Header from "@/components/Header";
import VideoFeed from "@/components/VideoFeed";
import TranscriptionPanel from "@/components/TranscriptionPanel";
import FaceRecognitionStatus from "@/components/FaceRecognitionStatus";
import { FaceDetection, Transcription, RecognitionStatus } from "@/utils/types";
import { toast } from "sonner";
import { fetchData, startSystem, stopSystem } from "@/utils/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const Index = () => {
  const [detectedFaces, setDetectedFaces] = useState<FaceDetection[]>([]);
  const [transcriptions, setTranscriptions] = useState<Transcription[]>([]);
  const [status, setStatus] = useState<RecognitionStatus>({
    isDetecting: false,
    isRecording: false,
    isProcessing: false,
    lastUpdated: Date.now(),
  });
  const [backendUrl, setBackendUrl] = useState<string>("http://localhost:5000");
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [hasError, setHasError] = useState<boolean>(false);

  // Function to fetch data from the backend
  const fetchBackendData = useCallback(async () => {
    if (!isConnected) return;

    try {
      const data = await fetchData(backendUrl);
      setDetectedFaces(data.faces);
      setTranscriptions(data.transcriptions);
      setStatus(data.status);
      setHasError(false);
    } catch (error) {
      console.error("Error fetching data:", error);
      setHasError(true);

      // Don't show toast on every polling failure, only on the first one
      if (!hasError) {
        toast.error("Connection error", {
          description: "Failed to fetch data from backend",
        });
      }
    }
  }, [isConnected, backendUrl, hasError]);

  // Connect to the backend
  const connectToBackend = async () => {
    if (!backendUrl.trim()) {
      toast.error("Invalid URL", {
        description: "Please enter a valid backend URL",
      });
      return;
    }

    setIsLoading(true);

    try {
      // Try to fetch initial data to verify connection
      await fetchData(backendUrl);

      // Start the system
      await startSystem(backendUrl);

      setIsConnected(true);
      setHasError(false);
      toast.success("Connected to backend", {
        description: `Successfully connected to ${backendUrl}`,
      });
    } catch (error) {
      console.error("Connection error:", error);
      toast.error("Connection failed", {
        description:
          "Could not connect to the backend. Make sure the server is running.",
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Disconnect from the backend
  const disconnectFromBackend = async () => {
    if (!isConnected) return;

    setIsLoading(true);

    try {
      await stopSystem(backendUrl);
      setIsConnected(false);
      toast.info("Disconnected", {
        description: "Successfully disconnected from the backend",
      });
    } catch (error) {
      console.error("Disconnect error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  // Poll for data from the backend
  useEffect(() => {
    if (!isConnected) return;

    // Initial fetch
    fetchBackendData();

    // Set up polling
    const interval = setInterval(fetchBackendData, 1000);

    return () => clearInterval(interval);
  }, [isConnected, fetchBackendData]);

  // Simulating API data for demo purposes when not connected
  useEffect(() => {
    // Only run the demo if not connected to a real backend
    if (isConnected) return;

    // Simulate face detection after a delay
    const faceDetectionTimer = setTimeout(() => {
      setStatus((prev) => ({
        ...prev,
        isDetecting: true,
        lastUpdated: Date.now(),
      }));

      // Add a face detection
      const demoFace: FaceDetection = {
        id: "1",
        name: "John Doe",
        confidence: 0.92,
        x: 40,
        y: 15,
        width: 25,
        height: 25,
      };

      setDetectedFaces([demoFace]);

      toast.success("Face detected", {
        description: "Successfully recognized John Doe",
      });

      // Simulate recording starting
      setTimeout(() => {
        setStatus((prev) => ({
          ...prev,
          isRecording: true,
          lastUpdated: Date.now(),
        }));

        // Simulate processing after recording
        setTimeout(() => {
          setStatus((prev) => ({
            ...prev,
            isRecording: false,
            isProcessing: true,
            lastUpdated: Date.now(),
          }));

          // Add transcription after "processing"
          setTimeout(() => {
            const newTranscription: Transcription = {
              text: "Hello, this is a demo of the face and speech recognition system. The detected speech would appear here in real-time.",
              timestamp: Date.now(),
              personName: "John Doe",
            };

            setTranscriptions((prev) => [...prev, newTranscription]);
            setStatus((prev) => ({
              ...prev,
              isProcessing: false,
              lastUpdated: Date.now(),
            }));

            toast.info("Speech transcribed", {
              description: "New speech transcription added",
            });
          }, 2000);
        }, 3000);
      }, 2000);
    }, 2000);

    // Add a second face and transcription after a longer delay
    const secondFaceTimer = setTimeout(() => {
      const secondFace: FaceDetection = {
        id: "2",
        name: "Visitor",
        confidence: 0.65,
        x: 65,
        y: 20,
        width: 20,
        height: 20,
      };

      setDetectedFaces((prev) => [...prev, secondFace]);

      // Add another transcription
      setTimeout(() => {
        setStatus((prev) => ({
          ...prev,
          isRecording: true,
          lastUpdated: Date.now(),
        }));

        setTimeout(() => {
          setStatus((prev) => ({
            ...prev,
            isRecording: false,
            isProcessing: true,
            lastUpdated: Date.now(),
          }));

          setTimeout(() => {
            const newTranscription: Transcription = {
              text: "In a real application, this would connect to your Python backend that processes video and audio data using machine learning models.",
              timestamp: Date.now(),
              personName: "John Doe",
            };

            setTranscriptions((prev) => [...prev, newTranscription]);
            setStatus((prev) => ({
              ...prev,
              isProcessing: false,
              lastUpdated: Date.now(),
            }));
          }, 2000);
        }, 2000);
      }, 1000);
    }, 8000);

    // Cleanup timers
    return () => {
      clearTimeout(faceDetectionTimer);
      clearTimeout(secondFaceTimer);
    };
  }, [isConnected]);

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      <Header />

      <main className="flex-grow container mx-auto px-6 py-8 max-w-7xl">
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Python Backend Connection</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col sm:flex-row items-center gap-2">
              <Input
                type="text"
                value={backendUrl}
                onChange={(e) => setBackendUrl(e.target.value)}
                placeholder="http://localhost:5000"
                className="flex-grow"
                disabled={isConnected || isLoading}
              />
              {isConnected ? (
                <Button
                  onClick={disconnectFromBackend}
                  variant="destructive"
                  disabled={isLoading}
                >
                  Disconnect
                </Button>
              ) : (
                <Button onClick={connectToBackend} disabled={isLoading}>
                  Connect
                </Button>
              )}
            </div>
            {isConnected && (
              <p className="text-sm text-green-600 mt-2">
                ✓ Connected to Python backend. Face recognition system is
                active.
              </p>
            )}
            {!isConnected && (
              <p className="text-sm text-slate-500 mt-2">
                Running in demo mode. Connect to your Python backend to use real
                face recognition.
              </p>
            )}
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-6">
            <VideoFeed
              detectedFaces={detectedFaces}
              isActive={isConnected}
              backendUrl={backendUrl}
            />
            <FaceRecognitionStatus status={status} />
          </div>

          {/* Right sidebar for transcriptions */}
          <div className="lg:col-span-1 h-[calc(100vh-10rem)]">
            <TranscriptionPanel
              transcriptions={transcriptions}
              isRecording={status.isRecording}
              className="h-full"
            />
          </div>
        </div>
      </main>

      <footer className="py-4 px-6 text-center text-xs text-slate-500 border-t border-slate-200/70">
        <p>
          Moratuwa Intelligent Robot (MIRob): An Intelligent Service Robot for
          Domestic Environments
        </p>
      </footer>
    </div>
  );
};

export default Index;
