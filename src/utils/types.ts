
export interface FaceDetection {
  id: string;
  name: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Transcription {
  text: string;
  timestamp: number;
  personName: string;
}

export interface RecognitionStatus {
  isDetecting: boolean;
  isRecording: boolean;
  isProcessing: boolean;
  lastUpdated: number;
}
