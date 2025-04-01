
/**
 * API service for communicating with the Python backend
 */

import { FaceDetection, Transcription, RecognitionStatus } from './types';

export interface ApiResponse {
  faces: FaceDetection[];
  transcriptions: Transcription[];
  status: RecognitionStatus;
}

/**
 * Fetches all data from the backend API
 */
export const fetchData = async (backendUrl: string): Promise<ApiResponse> => {
  try {
    const response = await fetch(`${backendUrl}/api/data`);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    const data = await response.json();
    
    // Convert timestamp from backend (in ms) to JS Date.now() format if needed
    const status: RecognitionStatus = {
      ...data.status,
      lastUpdated: data.status.lastUpdated || Date.now()
    };
    
    return {
      faces: data.faces || [],
      transcriptions: data.transcriptions || [],
      status: status
    };
  } catch (error) {
    console.error("API fetch error:", error);
    throw error;
  }
};

/**
 * Gets the video stream URL from the backend
 */
export const getVideoStreamUrl = (backendUrl: string): string => {
  return `${backendUrl}/video_feed`;
};

/**
 * Starts the face recognition system
 */
export const startSystem = async (backendUrl: string): Promise<{ status: string }> => {
  try {
    const response = await fetch(`${backendUrl}/api/start`);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to start system:", error);
    throw error;
  }
};

/**
 * Stops the face recognition system
 */
export const stopSystem = async (backendUrl: string): Promise<{ status: string }> => {
  try {
    const response = await fetch(`${backendUrl}/api/stop`);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to stop system:", error);
    throw error;
  }
};