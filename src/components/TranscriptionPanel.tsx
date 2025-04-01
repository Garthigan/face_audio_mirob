
import React, { useRef, useEffect } from 'react';
import { Mic, MessageSquare, Clock } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Transcription } from '@/utils/types';

interface TranscriptionPanelProps {
  className?: string;
  transcriptions: Transcription[];
  isRecording?: boolean;
}

const TranscriptionPanel: React.FC<TranscriptionPanelProps> = ({ 
  className, 
  transcriptions = [],
  isRecording = false
}) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new transcriptions arrive
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [transcriptions.length]);

  // Format timestamp to HH:MM:SS
  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit',
      hour12: false 
    });
  };

  return (
    <div className={cn(
      'transcription-panel flex flex-col h-full rounded-xl bg-white shadow-sm border border-slate-200/70 overflow-hidden animate-slide-up',
      className
    )}>
      <div className="p-4 border-b border-slate-200/70 flex items-center justify-between">
        <div className="flex items-center">
          <MessageSquare className="h-4 w-4 text-blue-500 mr-2" />
          <h3 className="font-medium text-sm">Speech Transcriptions</h3>
        </div>
        <div className="flex items-center">
          {isRecording && (
            <div className="flex items-center mr-3 px-2 py-0.5 rounded-full bg-red-50 text-red-500 text-xs font-medium">
              <div className="h-1.5 w-1.5 rounded-full bg-red-500 mr-1.5 recording-indicator"></div>
              Recording
            </div>
          )}
          <button className="p-1.5 rounded-md hover:bg-slate-100 transition-colors">
            <Mic className="h-3.5 w-3.5 text-slate-600" />
          </button>
        </div>
      </div>

      <div className="flex-grow overflow-y-auto p-4 space-y-3 bg-slate-50/50">
        {transcriptions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center p-6">
            <div className="w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center mb-3">
              <Mic className="h-6 w-6 text-blue-500" />
            </div>
            <h3 className="text-sm font-medium text-slate-700">No transcriptions yet</h3>
            <p className="text-xs text-slate-500 mt-1 max-w-[240px]">
              Speech will appear here once detected from recognized faces.
            </p>
          </div>
        ) : (
          transcriptions.map((item, index) => (
            <div 
              key={index} 
              className="flex flex-col p-3 rounded-lg bg-white border border-slate-200/70 shadow-sm transition-all hover:shadow-md"
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-blue-100 text-blue-700">
                  {item.personName}
                </span>
                <div className="flex items-center text-xs text-slate-500">
                  <Clock className="h-3 w-3 mr-1" />
                  {formatTime(item.timestamp)}
                </div>
              </div>
              
              <p className="text-sm text-slate-700 leading-relaxed">{item.text}</p>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
};

export default TranscriptionPanel;
